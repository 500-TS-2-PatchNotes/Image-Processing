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
import joblib
import uuid

from firebase_functions.core import init
from firebase_functions import https_fn, storage_fn
from firebase_admin import initialize_app
from firebase_admin import firestore, storage, credentials

from PatchNotesLib.ColourExtractor import ColourExtractor
from PatchNotesLib.Colour2BacteriaRegressor import Regressor

cred = credentials.Certificate('./patchnotes-d3b06-93b5440cb302.json')
app = initialize_app(cred)

db = firestore.client(app)
bucket = storage.bucket(app=app)

print("[INFO] Firebase Functions initialized!")
global regressor
print("[INFO] Global regressor variable initialized!")

def fetch_training_images() -> list:
    """
    Queries the 'training_images' collection in Firebase Storage
    Each document contains a URLs to multiple training images for the same predicted level

    Returns:
        A list of ColourExtractor objects
        A list of levels corresponding to the ColourExtractor objects
    """
    images = []
    levels = []
    # Fetch the training images from Firebase Storage
    print("[INFO] Fetching training images from Firestore...")
    collection = db.collection('training_images').get()

    if not collection:
        print("[ERROR] No training images found!")
        return [], []
    
    print(f"[INFO] Found {len(collection)} training images!")
    for doc in collection:
        doc_name = doc.id
        doc_data = doc.to_dict()
        image_urls = doc_data.get('urls', [])
        
        level = int(doc.id.strip('L'))
        
        print(f"[INFO] Document ID: {doc_name}, Level: {level}")
        if level is not None and image_urls:
            for url in image_urls:
                # Download the image from the URL
                print(f"[INFO] Downloading image from URL: {url}")
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        with open(f"./tmp/training_image.jpg", "wb") as f:
                            f.write(response.content)
                            f.close()

                except Exception as e:
                    print(f"[ERROR] Error downloading image: {e}")
                    return [], []

                extractor = ColourExtractor(f"./tmp/training_image.jpg")
                levels.append(level)
                images.append(extractor)

        print(f"[INFO] {len(images)} extractors created!")
    
    return images, levels

def train_regressor() -> bool:
    """
    Initialize the training data and train the Regressor.
    """

    if not os.path.exists('./tmp/'):
        os.makedirs('./tmp/')

    # Check if the regressor is already trained
    if os.path.exists('./tmp/regressor.pkl'):
        print("[INFO] Regressor already trained!")
        return True

    print("[INFO] Fetching Training Data...")

    # Fetch the training images
    training_extractors, y_train = fetch_training_images()

    if training_extractors == []:
        print("[ERROR] No training images found!")
        return False

    # Extract the dominant colours from the training images
    X_train = []

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
    # Save the trained regressor to a file
    joblib.dump(regressor, './tmp/regressor.pkl')
    print("[INFO] Regressor saved to file!")

@init
def init_train_regressor() -> bool:
    return train_regressor()

@storage_fn.on_object_finalized(memory=1024)
def analyze_image(event: storage_fn.CloudEvent[storage_fn.StorageObjectData]) -> str:
    """
    Listens for new images uploaded to images/{user_id}/{image_id} in Firebase Storage.
      - Reads 'image_id', 'user_id'
      - Uses ColourExtractor to get dominant colour (hex) from the image
      - Uses Regressor to predict a level
      - Determines a status (healthy if level < 2, unhealthy otherwise)
      - Stores the data in Firestore under the 'wound_data' collection
    """
    bucket_name = event.data.bucket
    filepath = event.data.name
    content_type = event.data.content_type

    print(f"[INFO] File {filepath} uploaded to bucket {bucket_name}.")

    if not content_type or not content_type.startswith("image/"):
        print(f"[INFO] This is not an image. ({content_type}) Returning...")
        return
    
    if not filepath.startswith("images/"):
        print(f"[INFO] File path does not start with 'images/'. ({filepath}) Returning...")
        return

    try:
        user_id = filepath.split('/')[1]
        image_id = filepath.split('/')[2]
    except Exception as e:
        print(f"[INFO] Error parsing image path. Likely not uploaded to the correct path format, or is not related to the user. ({filepath})")
        return

    if image_id == '' or None:
        print("[INFO] Image was not uploaded correctly!")
        return
    
    print("[INFO] Starting analysis...")

    print(f"[INFO] User ID: {user_id}, Image ID: {image_id}")
    
    # load the trained regressor from the file
    if not os.path.exists('./tmp/regressor.pkl'):
        print("[INFO] Regressor not found! Training...")

        if not train_regressor():
            return "[ERROR] Regressor training failed."
        
        print("[INFO] Regressor trained successfully!")

    print("[INFO] Loading Regressor from file...")
    regressor = joblib.load('./tmp/regressor.pkl')
    print("[INFO] Regressor loaded!")

    print("[INFO] Downloading image...")
    # Download the image
    if not os.path.exists(f'./tmp/{user_id}'):
        os.makedirs(f'./tmp/{user_id}')

    blob = bucket.blob(filepath)
    image = blob.download_as_bytes()

    md = blob.metadata or {}
        
    with open(f"./tmp/wound_image.png", "wb") as f:
        f.write(image)
        f.close()

    print(f"[INFO] Image downloaded to ./tmp/wound_image.png!")

    print(f"[INFO] Getting Firebase Storage Download Token!")

    # Check if the token is already set
    if "firebaseStorageDownloadTokens" in md:
        print(f"[INFO] Token already exists: {md['firebaseStorageDownloadTokens']}")
        token = md["firebaseStorageDownloadTokens"]

    else:
        print("[INFO] No token found. Generating a new one.")
        # Generate a new token
        new_token = str(uuid.uuid4())
        md["firebaseStorageDownloadTokens"] = new_token
        blob.metadata = md
        blob.patch()  # Save the updated metadata
        print(f"[INFO] New token generated: {new_token}")
        token = new_token

    image_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name.replace("/","%2F")}/o/{filepath.replace('/','%2F')}?alt=media&token={token}"
    print(f"[INFO] Image URL: {image_url}")

    print("[INFO] Extracting dominant colour...")
    # Extract the dominant colour from the image
    extractor = ColourExtractor(f"./tmp/wound_image.png")

    dominant_colour = extractor.get_dominant_colours()[0]
    print(f"[INFO] Dominant Extracted! Dominant Colour: {dominant_colour}")

    print("[INFO] Predicting level...")
    # Predict the level using the Regressor
    X_test = dominant_colour
    # Reshape the input to match the expected shape for the regressor
    X_test = np.array(X_test).reshape(1, -1)
    print(f"[INFO] X_test: {X_test}")
    y_pred = float(regressor.predict(X_test)[0])
    print(f"[INFO] Predicted level: {y_pred}")

    # Determine the status
    status = 0 if y_pred < 3 else 1
    print(f"[INFO] Status: {status} = {'Healthy' if status == 0 else 'Unhealthy'}")

    # timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    db.collection('users').document(str(user_id)).collection('wound_data').add(
        document_id = image_id.strip('.jpg').strip('.png'),
        document_data = {
            'user_id': user_id,
            'image_name': image_id,
            'analyze_time': timestamp,
            'level': y_pred,
            'unhealthy': status,
            'URL' : image_url
        }
    )

    print("[INFO] Data stored in Firestore!")

    return f"[INFO] Analysis complete. Level: {y_pred}, Status: {status}"
