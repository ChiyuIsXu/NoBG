# -*- coding: utf-8 -*-
"""Path configuration"""
import os

__all__ = ["path_dict", "print_project_dir"]

# Project Root
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))

# Data directory
# # COCO dataset, 2017
# # Common Objects in Context (COCO) is a large-scale object detection, segmentation, and captioning dataset.
dataset_dir = project_root + r"\datasets"

# Model directory
model_dir = project_root + r"\models"

# Loss Function directory
loss_dir = project_root + r"\loss"

# Mertics directory
metrics_dir = project_root + r"\metrics"

# Log directory
log_dir = project_root + r"\logs"

# Result directory
result_dir = project_root + r"\results"

# Path Dictionary
path_dict = {}
path_dict["project_root"] = project_root
path_dict["model_dir"] = model_dir
path_dict["result_dir"] = result_dir


def print_project_dir():
    print("\033[1;34m")
    print("=" * 104)
    print("Project Root:", project_root)
    print("Dataset Root:", dataset_dir)
    print("Models Root:", model_dir)
    print("Results Root:", result_dir)
    print("=" * 104)
    print("\033[0m")


if __name__ == "__main__":
    print_project_dir()
