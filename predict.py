from classification import Classification

classification = Classification(n_classes=2,
                                sequence_size=100,
                                n_features=1,
                                model_dir="data/logs")

classification.predict(image_path="data/teste.tif",
                       predicted_path="data/teste_predicted.tif",
                       batch_size=500)
