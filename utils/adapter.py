from typing import TYPE_CHECKING

from ultralytics import YOLO

import utils.strings

if TYPE_CHECKING:
    from utils.pipeline import Pipeline


class Adapter:
    """Adapter that implements branching logic for YOLO architecture selection."""

    pipeline: "Pipeline"

    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def execute(self, heads: dict[str, float], skip_and_use: str | None) -> YOLO:
        """Execute the adapter process and return a YOLO model based on head scores."""
        self.pipeline.logger.info(
            "Analysing provided dataset to determine optimal heads"
        )

        # Calculate scores for each combination
        a_score = heads.get("P1", 0) + heads.get("P2", 0) + heads.get("P3", 0)
        b_score = heads.get("P2", 0) + heads.get("P3", 0) + heads.get("P4", 0)
        c_score = heads.get("P3", 0) + heads.get("P4", 0) + heads.get("P5", 0)

        if skip_and_use is not None:
            self.pipeline.logger.warning(
                f"No optimal architecture found, using provided {skip_and_use}"
            )
            return YOLO(model=skip_and_use)

        # Save the appropriate YAML to a file and return the model
        if a_score > b_score and a_score > c_score:
            self.pipeline.logger.info("Chosen type A (P1, P2, P3)")
            path = self.pipeline.persistence.save_file(
                "model.yaml", utils.strings.BRANCHING_A_YAML
            )
            return YOLO(model=path)
        elif b_score > a_score and b_score > c_score:
            self.pipeline.logger.info("Chosen type B (P2, P3, P4)")
            path = self.pipeline.persistence.save_file(
                "model.yaml", utils.strings.BRANCHING_B_YAML
            )
            return YOLO(model=path)
        elif c_score > a_score and c_score > b_score:
            self.pipeline.logger.info("Chosen type C (P3, P4, P5)")
            path = self.pipeline.persistence.save_file(
                "model.yaml", utils.strings.BRANCHING_C_YAML
            )
            return YOLO(model=path)
        else:
            self.pipeline.logger.warning("No optimal architecture found, using default")
            return YOLO(model="yolo11m.yaml")

    def __str__(self) -> str:
        """Return a string representation of the adapter."""
        return "adapter-branching"
