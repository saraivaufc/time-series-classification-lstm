from classification import Classification

classification = Classification(sequence_size=255,
                                n_features=1,
                                model_dir="data/logs")

classification.predict("data/mosaic.tif", "data/predicted.tif")