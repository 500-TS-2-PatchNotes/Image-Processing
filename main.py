import os
import sys
from pandas import DataFrame

from lib.ColourExtractor import *
from lib.data2img import *

def main():
    img_loc = sys.argv[1]
    extractor = ColourExtractor(img_loc)
    
    print(f"Dominant Colors: {extractor.get_dominant_colours()}".replace(", np.", " ").replace("(np.", "(").replace("uint8(", "").replace(") ", ", ").replace("))", ")"))
    print(f"Average Color: {extractor.get_average_colour()}".replace(", np.", " ").replace("(np.", "(").replace("int64(", "").replace(") ", ", ").replace("))", ")"))

    df = extractor.toDF()

    print(df.head())

if __name__ == "__main__":
    main()
