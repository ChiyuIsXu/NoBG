# -*- coding: utf-8 -*-
""" Devices for PyTorch: get_device_info """
import sys
import logging
import torch

__all__ = ["get_device_info"]


def get_device_info(allow_cpu=True):

    if not allow_cpu:
        if not torch.cuda.is_available():
            logging.error("CUDA is not available. Exiting program.")
            sys.exit()
    else:
        if not torch.cuda.is_available():
            logging.warning("CUDA is not available, using CPU instead.")

    selected_device = "cuda" if torch.cuda.is_available() else "cpu"

    if selected_device != "cuda":
        logging.info("Using device %s", selected_device)
    else:
        logging.info("Using CUDA device")
        print()
        print("\033[1;34m=" * 84)
        print(f"CUDA is available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA current device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        print("=" * 84, "\033[0m")

    return selected_device


if __name__ == "__main__":
    device = get_device_info(allow_cpu=False)
    print(f"Selected device: {device}")
