from ColourExtractor import *
from RGB2Image import *

def test_internal_methods(extractor : ColourExtractor):
    print("Dominant Colors:", extractor.get_dominant_colours())
    print("Average Color:", extractor.get_average_colour())
    # print("Pixel Colours:", extractor.get_pixel_colours())

def test_csv(extractor : ColourExtractor):
    extractor.toCSV()

    converter = CSV2Image("output.csv", "./output_images/output.png")
    converter.convert(extractor.get_dimensions())

def main():
    extractor = ColourExtractor("./test_images/testImage_2.jpg")

    test_internal_methods(extractor)
    test_csv(extractor)

if __name__ == "__main__":
    main()