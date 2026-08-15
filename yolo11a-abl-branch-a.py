from utils.pipeline import Pipeline

p = Pipeline(
    dataset_yaml_path="/data2/jd1/datasets/kitti/data.yaml",
    project="yolo-kitti",
    config={"device": "0, 1", "name": "YOLO11a-abl-branch-a-3"},
)

p.create_model(abl_branch="A")

p.train(epochs=100, batch=16, seed=3)

p.evaluate()
