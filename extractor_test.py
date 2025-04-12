import time
import os

from PatchNotesLib.ColourExtractor import *
from PatchNotesLib.data2img import *

def test_internal_methods(extractor : ColourExtractor):
    print(f"Dominant Colors: {extractor.get_dominant_colours()}".replace(", np.", " ").replace("(np.", "(").replace("uint8(", "").replace(") ", ", ").replace("))", ")"))
    print(f"Average Color: {extractor.get_average_colour()}".replace(", np.", " ").replace("(np.", "(").replace("int64(", "").replace(") ", ", ").replace("))", ")"))
    # print("Pixel Colours:", extractor.get_pixel_colours())

def test_csv(extractor : ColourExtractor, output_csv="./output_csv/output.csv", output_png="./output_images/output.png"):
    extractor.toCSV(output_csv)

    converter = csv2img(output_csv, output_png)
    converter.convert(extractor.get_dimensions())

def test(extractor : ColourExtractor, output_csv="./output.csv", output_png="./output_images/output.png"):
    test_internal_methods(extractor)
    test_csv(extractor, output_csv, output_png)

def main():
    if not os.path.exists("./output_csv"):
        os.makedirs("./output_csv")

    if not os.path.exists("./output_images"):
        os.makedirs("./output_images")

    rgb_extractors = []
    rgb_extractors.append(ColourExtractor("./test_images/1_000000.jpg")) # Black      000
    rgb_extractors.append(ColourExtractor("./test_images/2_0000ff.jpg")) # Blue       001
    rgb_extractors.append(ColourExtractor("./test_images/3_00ff00.jpg")) # Green      010
    rgb_extractors.append(ColourExtractor("./test_images/4_00ffff.jpg")) # Cyan       011

    rgb_extractors.append(ColourExtractor("./test_images/5_ff0000.jpg")) # Red        100
    rgb_extractors.append(ColourExtractor("./test_images/6_ff00ff.jpg")) # Magenta    101
    rgb_extractors.append(ColourExtractor("./test_images/7_ffff00.jpg")) # Yellow     110
    rgb_extractors.append(ColourExtractor("./test_images/8_ffffff.jpg")) # White      111
    
    rgb_extractors.append(ColourExtractor("./test_images/testImage_1.jpg")) # example image 1
    rgb_extractors.append(ColourExtractor("./test_images/testImage_2.jpg")) # example image 2
    rgb_extractors.append(ColourExtractor("./test_images/nathan.jpg")) # Large Image

    for i, extractor in enumerate(rgb_extractors):
        print(f"Test {i+1}: {extractor.get_image_path()}")

        start_time = time.perf_counter_ns()
        test(extractor, f"./output_csv/output_{i+1}.csv", f"./output_images/output_{i+1}.png")
        time_taken_ms = (time.perf_counter_ns() - start_time) / 1e6

        print()
        
        print(f"Pixel Count: {extractor.get_pixel_count()}")
        print(f"Time taken: {time_taken_ms} ms")

        print("\n")
        
if __name__ == "__main__":
    main()