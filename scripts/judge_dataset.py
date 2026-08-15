import math
import os
from typing import TypeAlias

import cv2
from tqdm import tqdm

FitnessScores: TypeAlias = tuple[float, float, float, float, float]


def calculate_fitness_scores(area: float) -> FitnessScores:

    def calculate_score(area: float, mean: float, stdev: float) -> float:
        """Calculate the score of an area given a mean + standard deviation
        :param area: The area to calculate the score of
        :param mean: The mean of the normal distribution
        :param stdev: The standard deviation of the normal distribution
        :return: The score of the area"""
        return math.exp(-(((area - mean) / stdev) ** 2))

    """Calculate the scores of an area for each head
    :param area: The area to calculate the scores of
    :return: The scores of the area for each head"""
    return (
        calculate_score(area, 16**2, 16**2),  # P1
        calculate_score(area, 32**2, 32**2),  # P2
        calculate_score(area, 64**2, 64**2),  # P3
        calculate_score(area, 96**2, 96**2),  # P4
        calculate_score(area, 128**2, 128**2),  # P5
    )


def test_get_annotation_areas(
    train_annotation_path: str, train_image_path: str, target_size: int = 640
) -> list[float]:
    areas: list[float] = []

    for annotation_file in tqdm(
        os.listdir(train_annotation_path),
        desc=f"Processing images ({train_annotation_path})",
    ):
        img_name = annotation_file.replace(".txt", "")

        for image_file in os.listdir(train_image_path):
            if img_name in image_file:
                with open(
                    f"{train_annotation_path}/{annotation_file}",
                    encoding="utf-8",
                ) as annotation:
                    im = cv2.imread(f"{train_image_path}/{image_file}")

                    image_height, image_width, _ = im.shape

                    # Ultralytics-style aspect-ratio preserving resize
                    scale = min(
                        target_size / image_width,
                        target_size / image_height,
                    )

                    for line in annotation.readlines():
                        (
                            _,
                            _,
                            _,
                            s_annotation_width,
                            s_annotation_height,
                        ) = line.split()

                        # YOLO bbox dimensions are normalized
                        annotation_width = float(s_annotation_width)
                        annotation_height = float(s_annotation_height)

                        # Convert normalized bbox to original pixel dimensions
                        bbox_width_px = annotation_width * image_width
                        bbox_height_px = annotation_height * image_height

                        # Convert to dimensions as seen by the model input
                        resized_width = bbox_width_px * scale
                        resized_height = bbox_height_px * scale

                        # Padding does not change bbox width/height,
                        # so it does not affect area
                        area = resized_width * resized_height

                        areas.append(area)

                break

    return areas
