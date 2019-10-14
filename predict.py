from classification import Classification

classification = Classification(n_classes=2,
                                sequence_size=100,
                                n_features=1,
                                model_dir="data/logs")

classification.predict("data/mosaic.tif", "data/predicted.tif")
