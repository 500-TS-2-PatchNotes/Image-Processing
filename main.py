"""
main.py
===
This file contains the main logic for the Firebase Functions.

Authors:
-----
Dominic Choi
    GitHub: [CarrotBRRR](https://github.com/CarrotBRRR)
"""
import requests
import numpy as np
from datetime import datetime

from firebase_functions.core import init
from firebase_functions import firestore_fn, https_fn
import firebase_admin
from firebase_admin import firestore, storage, credentials

from lib.ColourExtractor import ColourExtractor
from lib.Colour2BacteriaRegressor import Regressor

# Initialize Firebase Admin
if not firebase_admin._apps:
    firebase_admin.initialize_app()

db = firestore.client()

def fetch_training_images() -> list:
    """
    Queries the 'training_images' collection in Storage and fetches the images.
    
    Returns:
        A list of ColourExtractor objects
    """
    images = []
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix="training_images/")
    
    for i, blob in enumerate(blobs):
        # Download the image
        blob.download_to_filename(f"/tmp/training_image_{i}.png")
        images.append(ColourExtractor(f"/tmp/training_image_{i}.png"))

    return images

if __name__ == "__main__":
    """
    Initialize the training data and train the Regressor.
    """

    # Fetch the training images
    training_images = fetch_training_images()

    # Extract the dominant colours from the training images
    X_train = []
    y_train = [0, 1, 2, 3, 4, 5, 6]

    for i, extractor in enumerate(training_images):
        X_train.append(extractor.get_dominant_colours()[0])

    # Train the Regressor
    regressor = Regressor()
    regressor.train(np.array(X_train), np.array(y_train))


@firestore_fn.on_document_created('images/{user_id}/{image_id}')
def analyze_image(event: firestore_fn.Event[firestore_fn.DocumentSnapshot | None]) -> str:
    """
    Listens for new images uploaded to images/{user_id}/{image_id}.
      - Reads 'image_id', 'user_id'
      - Uses ColourExtractor to get dominant colour (hex) from the image
      - Uses Regressor to predict a level
      - Determines a status (healthy if level < 2, unhealthy otherwise)
      - Stores the data in Firestore under the 'wounds' collection
    """
    
    if event.data is None:
        return "No data provided."

    # Get the image path, user ID, and image name from the query parameters
    image_id = event.data['image_id']
    user_id = event.data['user_id']

    image_path = f"images/{user_id}/{image_id}"

    # Download the image
    bucket = storage.bucket()
    blob = bucket.blob(image_path)
    blob.download_to_filename(f"/tmp/{image_path}")

    # Extract the dominant colour from the image
    extractor = ColourExtractor(f"/tmp/{image_path}")
    dominant_colour = extractor.get_dominant_colours_hex()[0]

    # Predict the level using the Regressor
    X_test = np.array([dominant_colour])
    y_pred = regressor.predict(X_test)[0]

    # Determine the status
    status = 0 if y_pred < 2 else 1
    
    # Store the data in Firestore
    db.collection('wounds').add({
        'user_id': user_id,
        'image_name': image_id,
        'analyze_time': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        'level': y_pred,
        'unhealthy': status
    })

    return f"Analysis complete. Level: {y_pred}, Status: {status}"
