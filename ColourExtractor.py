import cv2
import numpy as np
from pandas import DataFrame

class ColourExtractor:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Image not found or unable to load.")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    
    def get_dominant_colours(self, k=3) -> list[tuple[int, int, int]]:
        """Extracts the dominant colours using K-means clustering."""
        pixels = self.image.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        counts = np.bincount(labels.flatten())
        sorted_colours = [tuple(centers[i]) for i in np.argsort(-counts)]
        
        return sorted_colours
    
    def get_average_colour(self) -> tuple[int, int, int]:
        """Calculates the average colour of the image."""
        avg_colour = np.mean(self.image, axis=(0, 1)).astype(int)
        return tuple(avg_colour)
    
    def convert_colour_space(self, code) -> np.ndarray:
        """Converts the image to a different colour space based on the given code."""
        converted_image = cv2.cvtcolour(self.image, code)
        return converted_image
    
    def get_pixel_colours(self) -> list[tuple[int, int, int]]:
        """Returns the colours of each pixel in the image."""
        return [tuple(pixel) for row in self.image for pixel in row]
    
    def get_dimensions(self) -> tuple[int, int]:
        """Returns the dimensions of the image."""
        return self.image.shape[:2]
    
    def toCSV(self, filename=f'./output.csv'):
        """Converts the image to a CSV file."""
        # get resolution
        height, width = self.get_dimensions()
        # get pixel colours
        pixels = self.get_pixel_colours()
        ps = []

        for pixel in pixels:
            ps.append(f"{pixel[0]},{pixel[1]},{pixel[2]}")

        # image_data = self.image.reshape(-1, self.image.shape[1] * 3)

        merged_data = [[f"#{r:02x}{g:02x}{b:02x}" for r, g, b in row] for row in self.image]

        df = DataFrame(merged_data)
        df.to_csv(filename, index=False)

        print(f"CSV file saved as {filename}.")

