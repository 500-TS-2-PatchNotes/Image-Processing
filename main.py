import os

from lib.ColourExtractor import *
from lib.data2img import *

def main():
    img_loc = input("Enter the location of the image: ")
    extractor = ColourExtractor(img_loc)
    
    df = extractor.toDF()

    df.head()

if __name__ == "__main__":
    main()