from classification import Classification

classification = Classification(n_classes=5,
                                sequence_size=50,
                                n_features=1,
                                model_dir="data/logs")

classification.predict(image_path="data/clipped_mosaic.tif",
                       predicted_path="data/clipped_predicted.tif",
                       batch_size=500)
