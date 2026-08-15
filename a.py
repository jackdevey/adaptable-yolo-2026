from utils.pipeline import Pipeline

p = Pipeline(
    dataset_yaml_path="/data2/jd1/datasets/landing-pad-detection-2-v4/data.yaml",
    config={"device": "0, 1", "name": "yolo11a-lpd2-3"},
)

p.create_model()

p.train(epochs=100, batch=16, seed=3)

p.evaluate()
