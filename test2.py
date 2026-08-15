from utils.pipeline import Pipeline

p = Pipeline(
    dataset_yaml_path="/data2/jd1/datasets/humancar_50m/data.yaml",
    config={"device": "0, 1", "name": "tests"},
)

heads = p.dmanager.judge_heads()

print(heads)

test = p.adapter.execute(heads, None)
