from utils.pipeline import Pipeline

p = Pipeline(
    dataset_yaml_path="/data2/jd1/datasets/kitti/data.yaml",
    project="yolo-kitti",
    config={"device": "0, 1", "name": "yolo11a-kitti-1"},
)

p.create_model()

p.train(epochs=100, batch=16, seed=1)

p.evaluate()
