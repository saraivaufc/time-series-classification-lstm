from classification import Classification

classification = Classification(n_classes=5,
                                sequence_size=50,
                                n_features=1,
                                model_dir="data/logs")

classification.train("data/samples.csv",
                     epochs=1000,
                     batch_size=255)
