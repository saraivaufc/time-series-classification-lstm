from classification import Classification

classification = Classification(n_classes=2,
                                sequence_size=255,
                                n_features=1,
                                model_dir="data/logs")

classification.train("data/allSamples.csv")
