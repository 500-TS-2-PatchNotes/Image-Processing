import lib.ColourExtractor as ce
import lib.data2img as d2i
import lib.Colour2BacteriaRegressor as c2b

def main():
    extractors = []
    extractors.append(ce.ColourExtractor("./ReferenceColours/L0.png"))
    extractors.append(ce.ColourExtractor("./ReferenceColours/L1.png"))
    extractors.append(ce.ColourExtractor("./ReferenceColours/L2.png"))
    extractors.append(ce.ColourExtractor("./ReferenceColours/L3.png"))
    extractors.append(ce.ColourExtractor("./ReferenceColours/L4.png"))
    extractors.append(ce.ColourExtractor("./ReferenceColours/L5.png"))
    extractors.append(ce.ColourExtractor("./ReferenceColours/L6.png"))

    X_train = []
    y_train = [0, 1, 2, 3, 4, 5, 6] # =<2 is healthy range
    for i, extractor in enumerate(extractors):
        print(f"0: {extractor.get_dominant_colours()[0]})")
        print(f"1: {extractor.get_dominant_colours()[1]})")
        print()
        X_train.append(extractor.get_dominant_colours()[0])

    regressor = c2b.Regressor()
    regressor.train(X_train, y_train)

    test_imgs = []

    test_imgs.append(ce.ColourExtractor("./ReferenceColours/L0.png"))
    test_imgs.append(ce.ColourExtractor("./ReferenceColours/L1.png"))
    test_imgs.append(ce.ColourExtractor("./ReferenceColours/L2.png"))
    test_imgs.append(ce.ColourExtractor("./ReferenceColours/L3.png"))
    test_imgs.append(ce.ColourExtractor("./ReferenceColours/L4.png"))
    test_imgs.append(ce.ColourExtractor("./ReferenceColours/L5.png"))
    test_imgs.append(ce.ColourExtractor("./ReferenceColours/L6.png"))

    for i, test_img in enumerate(test_imgs):
        X_test = test_img.get_dominant_colours()[0]
        print(f"Predicting {X_test} (L{i}): ")
        y_pred = regressor.predict([X_test])
        print(y_pred)
        print()
    
if __name__ == "__main__":
    main()