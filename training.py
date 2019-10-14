from classification import Classification

classification = Classification(n_classes=2,
                                sequence_size=100,
                                n_features=1,
                                model_dir="data/logs")

classification.train("data/samples.csv",
                     epochs=34,
                     batch_size=255)
