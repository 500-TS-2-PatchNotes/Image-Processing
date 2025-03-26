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
import cv2

from firebase_functions import https_fn
import numpy as np
import firebase_admin
from firebase_admin import firestore

# Import your custom modules
from lib.ColourExtractor import ColourExtractor
from lib.Colour2BacteriaRegressor import Regressor

# Initialize Firebase Admin (only once)
if not firebase_admin._apps:
    firebase_admin.initialize_app()

db = firestore.client()

def fetch_training_images():
    """
    Queries the 'training_images' collection in Firestore, retrieves each document,
    and downloads the image from the URL stored in the 'image_url' field.
    
    Returns:
        A list of tuples: (document_id, image as a cv2 image)
    """
    images = []
    training_images_ref = db.collection("training_images")
    docs = training_images_ref.stream()
    
    for doc in docs:
        data = doc.to_dict()
        image_url = data.get("image_url")
        if image_url:
            try:
                # Download the image using HTTP GET
                response = requests.get(image_url)
                response.raise_for_status()  # Raise error for bad status
                # Convert the image data to a numpy array
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                # Decode the array into an OpenCV image (BGR by default)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if image is not None:
                    images.append((doc.id, image))
            except Exception as e:
                print(f"Error fetching image from {image_url}: {e}")
    return images

# Example usage:
if __name__ == "__main__":
    training_images = fetch_training_images()
    print(f"Fetched {len(training_images)} training images.")

@https_fn.on_request()
def analyze_image(req: https_fn.Request) -> https_fn.Response:
    """
    HTTPS function that:
      - Reads 'image_path', 'user_id', and 'image_name' from query parameters.
      - Uses ColourExtractor to get dominant colour (hex) from the image.
      - Uses Regressor to predict a level (dummy data used here).
      - Determines a status (healthy if level < 2, unhealthy otherwise).
      - Stores the data in Firestore under the 'wounds' collection.
    """
    # Retrieve required query parameters
    image_path = req.args.get("image_path")
    user_id = req.args.get("user_id")
    image_name = req.args.get("image_name")
    
    if not all([image_path, user_id, image_name]):
        return https_fn.Response("Missing one or more required parameters: image_path, user_id, image_name", status=400)
    
    try:
        # Use ColourExtractor to get dominant colours (hex)
        extractor = ColourExtractor(image_path)
        dominant_colours = extractor.get_dominant_colours_hex(k=3)
        # Pick the most dominant colour
        dominant_colour = dominant_colours[0] if dominant_colours else None

        # Use Regressor for prediction
        regressor = Regressor()

        # Get training data from Firestore
        X_train = []
        y_train = [0, 1, 2, 3, 4, 5, 6]  # =<2 is healthy range



        # Dummy test data: predict level based on an average-like value
        prediction = regressor.predict(np.array([[0.55, 0.55, 0.55]]))
        # Use the first predicted value and cast to int
        predicted_level = int(round(prediction[0]))
        
        # Determine status: healthy if predicted_level < 2, unhealthy otherwise
        status = predicted_level >= 2  # True if unhealthy
        
        # Build the data object to store in Firestore
        wound_data = {
            "user_id": user_id,
            "image_name": image_name,
            "predicted_level": predicted_level,
            "dominant_colour": dominant_colour,
            "status": status
        }
        
        # Store the data in the "wounds" collection
        doc_ref = db.collection("wounds").add(wound_data)
        
        # Optionally, get the auto-generated document id:
        doc_id = doc_ref[1].id
        
        response_payload = {
            "wound_document_id": doc_id,
            "data_stored": wound_data
        }
        
        return https_fn.Response(str(response_payload), status=200)
    
    except Exception as e:
        return https_fn.Response(f"Error: {str(e)}", status=500)
