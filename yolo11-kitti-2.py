from utils.pipeline import Pipeline

p = Pipeline(
    dataset_yaml_path="/data2/jd1/datasets/kitti/data.yaml",
    project="yolo-kitti",
    config={"device": "0, 1", "name": "yolo11-kitti-2"},
)

p.create_model(skip_and_use="yolo11m.yaml")

p.train(epochs=100, batch=16, seed=2)

p.evaluate()
