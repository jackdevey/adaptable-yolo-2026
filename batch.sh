#!/bin/bash
set -e

python yolo11a-kitti-1.py
python yolo11-kitti-1.py
python yolo11a-kitti-2.py
python yolo11-kitti-2.py
python yolo11a-kitti-3.py
python yolo11-kitti-3.py
