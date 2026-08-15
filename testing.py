from utils.pipeline import Pipeline

print("OLD")

p = Pipeline(
    dataset_yaml_path="/data2/jd1/datasets/sds/compressed_yolo/data.yaml",
    project="testing",
    config={"device": "0, 1", "name": "testing"},
)

heads = p.dmanager.judge_heads()

print(heads)

test = p.adapter.execute(heads, None, None)

print("NEW")

from scripts.judge_dataset import calculate_fitness_scores, test_get_annotation_areas

areas = test_get_annotation_areas(
    train_annotation_path="/data2/jd1/datasets/sds/compressed_yolo/labels/train",
    train_image_path="/data2/jd1/datasets/sds/compressed_yolo/images/train",
    target_size=640,
)

# Count scores for each area
p5score = p4score = p3score = p2score = p1score = 0.0
# Open the ground truth files
for area in areas:
    scores = calculate_fitness_scores(area)

    p1score += scores[0]
    p2score += scores[1]
    p3score += scores[2]
    p4score += scores[3]
    p5score += scores[4]

# Return the scores for each head
heads = {
    "P1": p1score,
    "P2": p2score,
    "P3": p3score,
    "P4": p4score,
    "P5": p5score,
}

print(heads)

test = p.adapter.execute(heads, None, None)
