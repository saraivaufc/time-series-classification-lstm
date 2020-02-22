from classification import Classification

classification = Classification(n_classes=2,
                                sequence_size=30,
                                n_features=1,
                                model_dir="data/logs")

classification.train("data/2018-01-01_2019-12-31_222_74.csv",
                     epochs=1000,
                     batch_size=255)
