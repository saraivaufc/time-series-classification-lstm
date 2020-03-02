from classification import Classification

filename = "2018-01-01_2019-12-31_222_72.tif"

classification = Classification(n_classes=2,
                                sequence_size=50,
                                n_features=1,
                                model_dir="data/logs")

classification.predict(image_path="data/mosaics/{f}".format(f=filename),
                       predicted_path="data/predicted/{f}".format(f=filename),
                       batch_size=1024)

