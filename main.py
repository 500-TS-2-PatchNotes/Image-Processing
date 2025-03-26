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

from firebase_functions import https_fn
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


@https_fn.on_request()
def analyze_image(req: https_fn.Request) -> https_fn.Response:
    """
    HTTPS function that:
      - Reads 'image_path', 'user_id', and 'image_name' from query parameters.
      - Uses ColourExtractor to get dominant colour (hex) from the image.
      - Uses Regressor to predict a level.
      - Determines a status (healthy if level < 2, unhealthy otherwise).
      - Stores the data in Firestore under the 'wounds' collection.
    """
    # ???
    image_path = req.args.get('image_path')
    user_id = req.args.get('user_id')
    image_name = req.args.get('image_name')

    # Download the image
    response = requests.get(image_path)
    with open(f"/tmp/{image_name}", "wb") as f:
        f.write(response.content)

    # Extract the dominant colour
    extractor = ColourExtractor(f"/tmp/{image_name}")
    X_test = extractor.get_dominant_colours()[0]

    # Predict the level
    y_pred = regressor.predict([X_test])[0]

    # Determine the status
    status = y_pred <= 2

    # Store the data in Firestore
    db.collection('wounds').add({
        'user_id': user_id,
        'image_name': image_name,
        'analyze_time': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        'level': y_pred,
        'unhealthy': status
    })

    return f"Analysis complete. Level: {y_pred}, Status: {status}"
