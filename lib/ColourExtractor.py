"""
ColourExtractor
=====
A class that extracts colour information from an image


Authors:
-----
Dominic Choi
    GitHub: [CarrotBRRR](https://github.com/CarrotBRRR)

Initialization:
----
`ColourExtractor(image_path)`

Methods:
-----
- `get_dominant_colours(k=3) -> list[tuple[int, int, int]]`: Extracts the dominant colours using K-means clustering.
- `get_average_colour() -> tuple[int, int, int]`: Calculates the average colour of the image.

- `get_pixel_colour(x, y) -> tuple[int, int, int]`: Returns the colour of the pixel at the given coordinates.
- `get_pixel_colours() -> list[tuple[int, int, int]]`: Returns the colours of each pixel in the image.

- `get_dimensions() -> tuple[int, int]`: Returns the dimensions of the image.
- `get_pixel_count() -> int`: Returns the number of pixels in the image.

- `get_image_path() -> str`: Returns the path of the image.
- `toCSV(filename=f'./output.csv')`: Converts the image to a CSV file.

Created: 2025-02-18
"""

import cv2
import numpy as np
from pandas import DataFrame

class ColourExtractor:
    def __init__(self, image_path):
        """Initialises the ColourExtractor object with the given image path."""
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Image not found or unable to load.")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    
    def get_dominant_colours(self, k=3) -> list[tuple[int, int, int]]:
        """Extracts the dominant colours using K-means clustering and returns a list of tuples of [R, G, B]."""
        pixels = self.image.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        counts = np.bincount(labels.flatten())
        sorted_colours = [tuple(centers[i]) for i in np.argsort(-counts)]
        
        return sorted_colours
    
    def get_average_colour(self) -> tuple[int, int, int]:
        """Calculates the average colour of the image and returns a tuple of [R, G, B]."""
        avg_colour = np.mean(self.image, axis=(0, 1)).astype(int)
        return tuple(avg_colour)
    
    def convert_colour_space(self, code) -> np.ndarray:
        """Converts the image to a different colour space based on the given code."""
        converted_image = cv2.cvtcolour(self.image, code)
        return converted_image
    
    def get_pixel_colour(self, x, y) -> tuple[int, int, int]:
        """Returns the colour of the pixel at the given coordinates as a tuple of [R, G, B]."""
        return tuple(self.image[y, x])

    def get_pixel_colours(self) -> list[tuple[int, int, int]]:
        """Returns the colours of each pixel in the image as a list of tuples of [R, G, B]."""
        return [tuple(pixel) for row in self.image for pixel in row]
    
    def get_dimensions(self) -> tuple[int, int]:
        """Returns the dimensions of the image as a tuple of [width, height]."""
        return self.image.shape[:2]
    
    def get_pixel_count(self) -> int:
        """Returns the number of pixels in the image as an int."""
        x, y = self.get_dimensions()
        return x * y
    
    def get_image_path(self) -> str:
        """Returns the path of the image as a string."""
        return self.image_path

    def toCSV(self, filename=f'./output.csv'):
        """Converts the image to a CSV file."""
        rgb_data = [[f"#{r:02x}{g:02x}{b:02x}" for r, g, b in row] for row in self.image]

        df = DataFrame(rgb_data)
        df.to_csv(filename, index=False)

        print(f"CSV file saved as {filename}")

    def toDF(self):
        """Converts the image to a DataFrame."""
        rgb_data = [[f"#{r:02x}{g:02x}{b:02x}" for r, g, b in row] for row in self.image]

        return DataFrame(rgb_data)