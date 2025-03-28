"""
data2img
=====
Classes to convert raw data to images

Authors:
-----
Dominic Choi
    GitHub: [CarrotBRRR](https://github.com/CarrotBRRR)

Classes:
-----
- csv2img: Converts a CSV file of hex colours to a PNG image.
- *NOT WORKING*: df2img: Converts a DataFrame of rgb values to an image.

Initialization:
----
`CSV2Image(csv_path, output_png)`

`df2img(df, output_png)`

Created: 2025-02-18
"""

import pandas as pd
import numpy as np
import cv2

class csv2img:
    def __init__(self, csv_path:str, output_png:str):
        self.csv_path = csv_path
        self.output_png = output_png
    
    def hex_to_rgb(self, hex_color: str) -> tuple:
        """Converts a hex color string to an RGB tuple."""
        hex_color = hex_color.replace("#", "")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def convert(self, dimensons: tuple = (1, 1)):
        """Reads the CSV file, converts hex colors to RGB, and saves as a PNG image."""
        df = pd.read_csv(self.csv_path, header=None, dtype=str)
        df = df.drop(0)  # remove first row (index)

        # Convert hex values to RGB tuples
        rgb_array = np.array(df)
        rgb_data = np.array([self.hex_to_rgb(hex_color) for hex_color in rgb_array.flatten()])

        rgb_data = rgb_data.reshape(int(dimensons[0]), int(dimensons[1]), 3)    # Reshape into 2D image array
        bgr_data = cv2.cvtColor(rgb_data.astype(np.uint8), cv2.COLOR_RGB2BGR)   # Convert to BGR

        # Save as PNG
        cv2.imwrite(self.output_png, bgr_data)
        print(f"PNG saved to {self.output_png}")

class df2img:
    def __init__(self, df:pd.DataFrame, output_png):
        self.df = df
        self.output_png = output_png

    def convert(self):
        """Converts a DataFrame to an image."""
        rgb_data = np.array(self.df)
        rgb_data = rgb_data.reshape(rgb_data.shape[0], rgb_data.shape[1], 3)    # Reshape into 2D image array
        bgr_data = cv2.cvtColor(rgb_data.astype(np.uint8), cv2.COLOR_RGB2BGR)   # Convert to BGR

        # Save as PNG
        cv2.imwrite(self.output_png, bgr_data)
        print(f"PNG saved to {self.output_png}")