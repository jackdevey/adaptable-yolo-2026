from utils.pipeline import Pipeline

p = Pipeline(
    dataset_yaml_path="/data2/jd1/datasets/landing-pad-detection-2-v4/data.yaml",
    config={"device": "0, 1", "name": "tests"},
)

p.load_model(
    path="/data2/jd1/working/adaptable-yolo-2026/.yolo-out/yolo11-lpd2-v4-4/weights/epoch50.pt"
)


p.evaluate()
