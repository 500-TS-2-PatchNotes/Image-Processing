import pandas as pd
import numpy as np
import cv2


class CSV2Image:
    def __init__(self, csv_path, output_png):
        self.csv_path = csv_path
        self.output_png = output_png
    
    def hex_to_rgb(self, hex_color: str) -> tuple:
        """Converts a hex color string to an RGB tuple."""
        hex_color = hex_color.replace("#", "")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def convert(self, dimensons: tuple = (1, 1)):
        """Reads the CSV file, converts hex colors to RGB, and saves as a PNG image."""
        df = pd.read_csv(self.csv_path, header=None, dtype=str)
        indices = df[0]
        df = df.drop(0)  # remove first row (index)
        print(df.columns)

        # Convert hex values to RGB tuples
        rgb_array = np.array(df)
        rgb_data = np.array([self.hex_to_rgb(hex_color) for hex_color in rgb_array.flatten()])
    
        print(rgb_data)
        print(rgb_data.shape)

        # Reshape into 2D image 
        rgb_data = rgb_data.reshape(int(dimensons[0]), int(dimensons[1]), 3)    

        

        # Save as PNG
        cv2.imwrite(self.output_png, rgb_data)
        print(f"PNG saved to {self.output_png}")


# Example usage:
# converter = CSVToPNGConverter("path/to/colors.csv", "output.png")
# converter.convert()
