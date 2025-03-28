"""
main.py
===
This file contains the main logic for the Firebase Functions.

Authors:
-----
Dominic Choi
    GitHub: [CarrotBRRR](https://github.com/CarrotBRRR)
"""
import os
import requests
import numpy as np
from datetime import datetime

from firebase_functions.core import init
from firebase_functions import https_fn, storage_fn
import firebase_admin
from firebase_admin import firestore, storage, credentials

from lib.ColourExtractor import ColourExtractor
from lib.Colour2BacteriaRegressor import Regressor


app = firebase_admin.initialize_app()

db = firestore.client(app)
bucket = storage.bucket(app)

global regressor
regressor = None

def fetch_training_images() -> list:
    """
    Queries the 'training_images' collection in Firebase Storage

    Returns:
        A list of ColourExtractor objects
    """
    images = []

    # Fetch the training images from Firebase Storage
    print("[INFO] Fetching training images from Firestore...")
    collection = db.collection('training_images').get()

    # Extract the image URLs from the documents
    image_urls = [doc.to_dict()['url'] for doc in collection]
    print(f"[INFO] Found {len(image_urls)} training image URLs!")

    print("[INFO] Downloading training images...")
    # Download each image and create a ColourExtractor object
    for image_url in image_urls:
        # Download the image
        response = requests.get(image_url)

        # Save the image to a temporary file
        with open('./tmp/temp_image.jpg', 'wb') as f:
            f.write(response.content)

        extractor = ColourExtractor('./tmp/temp_image.jpg')
        images.append(extractor)

    print(f"[INFO] Training image {len(images)}/{len(image_urls)} extractors created.")

    return images

@init
def train_regressor() -> bool:
    """
    Initialize the training data and train the Regressor.
    """

    if not os.path.exists('./tmp/'):
        os.makedirs('./tmp/')

    if regressor is not None:
        print("[INFO] Regressor already trained!")
        return False

    print("[INFO] Fetching Training Data...")

    # Fetch the training images
    training_extractors = fetch_training_images()

    if training_extractors == []:
        print("[ERROR] No training images found!")
        return False

    # Extract the dominant colours from the training images
    X_train = []
    y_train = [0, 1, 2, 3, 4, 5, 6]

    for extractor in training_extractors:
        X_train.append(extractor.get_dominant_colours()[0])

    print("[INFO] Training Data Fetched!")

    print("[INFO] Initializing Regressor...")
    # Train the Regressor
    regressor = Regressor()

    print("[INFO] Training Regressor...")
    regressor.train(np.array(X_train), np.array(y_train))
    print("[INFO] Regressor Trained!")

    print("[INFO] Regressor ready for use!")

    return True

@storage_fn.on_finalized(path="images/{user_id}/{image_id}")
def analyze_image(event: storage_fn.CloudEvent[storage_fn.StorageObjectData]) -> str:
    """
    Listens for new images uploaded to images/{user_id}/{image_id} in Firebase Storage.
      - Reads 'image_id', 'user_id'
      - Uses ColourExtractor to get dominant colour (hex) from the image
      - Uses Regressor to predict a level
      - Determines a status (healthy if level < 2, unhealthy otherwise)
      - Stores the data in Firestore under the 'wounds' collection
    """
    bucket_name = event.data.bucket
    filepath = event.data.name
    content_type = event.data.content_type

    print(f"[INFO] File {filepath} uploaded to bucket {bucket_name}.")
    print("[INFO] Starting analysis...")

    if not content_type or not content_type.startswith("image/"):
        print(f"This is not an image. ({content_type})")
        return
    
    user_id = filepath.split('/')[1]
    image_id = filepath.split('/')[2]
    print(f"[INFO] User ID: {user_id}, Image ID: {image_id}")

    # Check if the regressor is trained
    if regressor is None:
        print("[WARN] Regressor not trained!")
        # Attempt to train the regressor
        if not train_regressor():
            return "Regressor training failed."

        print("[INFO] Regressor trained successfully!")

        return "Regressor not trained."

    print("[INFO] Downloading image...")
    # Download the image
    blob = bucket.blob(filepath)
    blob.download_to_filename(f"/tmp/{filepath}")
    print("[INFO] Image downloaded!")

    print("[INFO] Extracting dominant colour...")
    # Extract the dominant colour from the image
    extractor = ColourExtractor(f"/tmp/{filepath}")
    dominant_colour = extractor.get_dominant_colours()[0]
    print(f"[INFO] Dominant Extracted! Dominant Colour: {dominant_colour}")

    print("[INFO] Predicting level...")
    # Predict the level using the Regressor
    X_test = dominant_colour
    y_pred = regressor.predict(X_test)[0]
    print(f"[INFO] Predicted level: {y_pred}")

    # Determine the status
    status = 0 if y_pred < 2 else 1
    print(f"[INFO] Status: {'Healthy' if status == 0 else 'Unhealthy'}")

    # Store the data in Firestore
    db.collection('wounds').add({
        'user_id': user_id,
        'image_name': image_id,
        'analyze_time': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        'level': y_pred,
        'unhealthy': status
    })

    return f"Analysis complete. Level: {y_pred}, Status: {status}"

@https_fn.on_request()
def http_test(request: https_fn.Request) -> https_fn.Response:
    """
    Test function for HTTP requests.
    """
    return https_fn.Response("Hello from Firebase Functions!")
