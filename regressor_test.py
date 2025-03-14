import lib.ColourExtractor as ce
import lib.data2img as d2i
import lib.Colour2BacteriaRegressor as c2b

def main():
    extractors = []
    extractors.append(ce.ColourExtractor("./ReferenceColours/L0.jpg"))
    extractors.append(ce.ColourExtractor("./ReferenceColours/L1.jpg"))
    extractors.append(ce.ColourExtractor("./ReferenceColours/L2.jpg"))
    extractors.append(ce.ColourExtractor("./ReferenceColours/L3.jpg"))
    extractors.append(ce.ColourExtractor("./ReferenceColours/L4.jpg"))
    extractors.append(ce.ColourExtractor("./ReferenceColours/L5.jpg"))
    extractors.append(ce.ColourExtractor("./ReferenceColours/L6.jpg"))

    X_train = []
    y_train = [0, 1, 2, 3, 4, 5, 6] # =<2 is healthy range
    for i, extractor in enumerate(extractors):
        X_train.append(extractor.get_dominant_colours()[1])

    regressor = c2b.Regressor()
    regressor.train(X_train, y_train)

    # Test Predict of Level 2, since it is the max considered healthy
    L2_img = ce.ColourExtractor("./ReferenceColours/L2.jpg")
    X_test = L2_img.get_dominant_colours()[1]
    print(f"predicting {X_test} (L2 healthy):")

    y_pred = regressor.predict([X_test])
    print(y_pred)

    # Test Predict of Level 3, since it is considered unhealthy
    L3_img = ce.ColourExtractor("./ReferenceColours/L3.jpg")
    X_test = L3_img.get_dominant_colours()[1]
    print(f"predicting {X_test} (L3 unhealthy):")
    
    y_pred = regressor.predict([X_test])
    print(y_pred)

    # Test Predict of a random image
    img_test = ce.ColourExtractor("./test_images/nathan.jpg")
    X_test = img_test.get_dominant_colours()[1]
    print(f"predicting {X_test} (nathan.jpg):")

    y_img_pred = regressor.predict([X_test])
    print(y_img_pred)

if __name__ == "__main__":
    main()