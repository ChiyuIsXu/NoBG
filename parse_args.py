# -*- coding: utf-8 -*-
""" Module to parse the arguments from the command line """

import os
import logging
import argparse
from enum import Enum
from typing import Dict

import yaml

__all__ = [
    "get_args",
    "Constants",
    "load_or_create_config",
]


# Constants
class Constants(Enum):
    """
    Constants class to store the names of
    the architectures, losses and optimizers.e.g
    """

    ARCH_NAMES = ["UNet", "NestedUNet"]
    LOSS_NAMES = ["BCEDiceLoss", "LovaszHingeLoss", "BCEWithLogitsLoss"]
    OPTIMEZER_NAMES = ["Adam", "SGD"]


def get_args():
    """Function to get the arguments from the command line"""
    parser = argparse.ArgumentParser(
        description="Train the UNet on \
            Mulit-Sequence CMR images and target masks"
    )
    parser.add_argument(
        "--name", default=None, help="model name (default: dataset_arch_with-DS-or-not)"
    )
    # random seed
    # What is the meaning of life, the universe, and everything?
    parser.add_argument("--seed", default=42, type=int)

    # Parameters for the model
    model = parser.add_argument_group("Model", "Parameters for the model")
    # model architecture
    model.add_argument(
        "--arch",
        metavar="ARCH",
        default="NestedUNet2D",
        choices=Constants.ARCH_NAMES.value[:],
        help="model architecture: "
        + " | ".join(Constants.ARCH_NAMES.value[:])
        + " (default: UNet2D)",
        dest="arch",
    )
    # load model
    model.add_argument(
        "--load", type=str, default=False, help="Load model from a .pth file"
    )
    # for multi-class segmentation, the number of classes should be specified
    model.add_argument(
        "--classes",
        "-c",
        type=int,
        default=2,
        help="Number of classes",
        dest="num_classes",
    )  # default is 2 classes: background and foreground.
    # our project has 6 classes: background, lv, rv, myo, edema, scar

    # training
    train = parser.add_argument_group("Train", "Parameters for the Training")
    train.add_argument(
        "--epochs",
        "-e",
        metavar="E",
        type=int,
        default=5,
        help="Number of total epochs to run",
        dest="epochs",
    )
    train.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        metavar="B",
        type=int,
        default=16,
        help="mini-batch size (default: 16)",
    )
    train.add_argument("--num_workers", default=0, type=int)
    train.add_argument(
        "--learning-rate",
        metavar="LR",
        type=float,
        default=1e-5,
        help="Learning rate",
        dest="learning_rate",
    )
    train.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Use mixed precision. \
            When enabled, some calculations will use \
            half-precision floating-point numbers (FP16),\
            while the remaining calculations will still use \
            single-precision floating-point numbers (FP32). \
            This can improve the speed and efficiency of model training, \
            especially when using modern Gpus.",
    )
    # 通常用于深度学习加速。
    # 启用后，某些计算会使用半精度浮点数（FP16）
    # 其余计算仍然使用单精度浮点数（FP32）。
    # 这可以提高模型训练的速度和效率，尤其是在使用现代GPU时。
    train.add_argument(
        "--deep_supervision",
        action="store_true",
        default=False,
        help="Use deep supervision. \
            When enabled, the model will output the final output \
            as well as intermediate outputs from the decoder. \
            This can improve the model's performance.",
    )
    # 使用深度监督。启用后，模型将输出最终输出以及解码器的中间输出。

    # loss
    loss = parser.add_argument_group("Loss", "Parameters for the Loss function")
    loss.add_argument(
        "--loss",
        default="BCEDiceLoss",
        choices=Constants.LOSS_NAMES.value[:],
        help="loss: "
        + " | ".join(Constants.LOSS_NAMES.value[:])
        + " (default: BCEDiceLoss)",
    )

    # optimizer
    optimizer = parser.add_argument_group("Optimizer", "Parameters for the Optimizer")
    optimizer.add_argument(
        "--optimizer",
        default="SGD",
        choices=Constants.OPTIMEZER_NAMES.value[:],
        help="optimizer: "
        + " | ".join(Constants.OPTIMEZER_NAMES.value[:])
        + " (default: Adam)",
    )

    optimizer.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        help="momentum. \
        This parameter is used for momentum optimizers \
        (such as SGD with momentum) \
        The momentum factor for accelerating the decline of the gradient.",
    )
    # 该参数用于动量优化器（如 SGD with momentum），
    # 表示用于加速梯度下降的动量因子

    optimizer.add_argument(
        "--nesterov",
        default=False,
        type=str2bool,
        help="nesterov. \
        This parameter is used for momentum optimizers \
        (such as SGD with momentum) \
        This helps prevent overfitting of the model.",
    )
    # 该参数用于指定是否使用 Nesterov 动量
    # type=str2bool 表示将命令行输入的字符串（如 “True”, “False”）转化为布尔值。
    optimizer.add_argument(
        "--weight_decay", default=1e-4, type=float, help="weight decay"
    )
    # 该参数用于控制权重衰减（L2 正则化）项的大小，这有助于防止模型过拟合。

    # data
    data = parser.add_argument_group("Data", "Parameters for the Data")
    # modality
    data.add_argument(
        "--shuffle",
        default=False,
        help="Use this argument to shuffle the data before processing",
    )
    data.add_argument(
        "--input_channels", default=3, type=int, help="Number of input channels"
    )
    data.add_argument("--modality", default=1, type=int, help="Number of modalities")
    # dataset
    data.add_argument("--dataset", default="Myops", help="dataset name")
    data.add_argument("--img_ext", default=".nii.gz", help="image file extension")
    data.add_argument("--mask_ext", default=".nii.gz", help="mask file extension")
    # validation
    data.add_argument(
        "--validation",
        dest="validation",
        type=float,
        default=0.0,
        help="Percent of the data that is used as validation (0-100)",
    )
    data.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Downscaling factor of the images",
    )
    data.add_argument(
        "--bilinear",
        action="store_true",
        default=False,
        help="Use bilinear upsampling. \
            If this parameter is specified, the program will be sampled using \
            a two-wire interpolation method. Otherwise, the program may use \
            other sampling methods (such as the recent interpolation). \
            The above sampling is a common step in image processing \
            and generation, especially in semantic segmentation, \
            generating antagonistic networks and other image processing tasks.",
    )
    # 如果该参数被指定，程序将使用双线性插值方法进行上采样；
    # 否则，程序可能会使用其他上采样方法（例如最近邻插值）。
    # 上采样是图像处理和生成中常见的一步，尤其是在语义分割、生成对抗网络和其他图像处理任务中。

    # scheduler
    scheduler = parser.add_argument_group("Scheduler", "Parameters for the Scheduler")
    scheduler.add_argument(
        "--scheduler",
        default="StepLR",
        choices=[
            "StepLR",
            "CosineAnnealingLR",
            "ReduceLROnPlateau",
            "MultiStepLR",
            "ConstantLR",
        ],
        help="Specifies which learning rate scheduler to use. \
            The learning rate scheduler is used to adjust the learning rate \
            during the training process to improve the training effect and \
            stability of the model",
    )
    # 该参数指定了使用哪种学习率调度器。
    # 学习率调度器用于在训练过程中调整学习率，以提高模型的训练效果和稳定性。
    scheduler.add_argument(
        "--min_lr",
        default=1e-5,
        type=float,
        help="minimum learning rate. \
            Some schedulers will gradually reduce the learning rate, \
            but will not be lower than this value.",
        dest="min_learning_rate",
    )
    # 设置训练过程中允许的最低学习率。
    # 某些调度器会逐渐减小学习率，但不会低于该值。

    scheduler.add_argument(
        "--factor",
        default=0.1,
        type=float,
        help="Used by the ReduceLROnPlateau scheduler. \
        Some schedulers will reduce the learning rate by this factor.",
    )
    # 用于ReduceLROnPlateau调度器，
    # 表示当监测指标停止提升时，学习率将乘以这个因子以减小。

    scheduler.add_argument(
        "--patience",
        default=2,
        type=int,
        help="Used by the ReduceLROnPlateau scheduler. \
        Some schedulers will reduce the learning rate after the monitored metric \
        stops improving for this many validation epochs. ",
    )
    # 用于 ReduceLROnPlateau 调度器，
    # 在降低学习率之前等待多少个验证周期
    # 如果验证指标在这段时间内没有改善，学习率将会被减少。

    scheduler.add_argument(
        "--milestones",
        default="1,2",
        type=str,
        help="milestones. \
        This parameter is a comma-separated list of integers that \
        represent the specific epochs at which to reduce the learning rate.",
    )
    scheduler.add_argument(
        "--step_size",
        default=3,
        type=int,
        help="step size. \
        This parameter is used by the StepLR scheduler \
        to reduce the learning rate by a factor of gamma every step_size epochs.",
    )
    # 该参数是用于 MultiStepLR 调度器，表示一个逗号分隔的整数列表
    # 表示在训练到这些具体的纪元时降低学习率
    scheduler.add_argument(
        "--gamma",
        default=2 / 3,
        type=float,
        help="gamma. \
        This parameter represents the multiplication factor when \
        reducing the learning rate at each milestone. \
        It is usually used for MultiStepLR schedulers \
        and CosineAnnealingLR schedulers.",
    )
    # 该参数表示在每个里程碑降低学习率时的乘法因子，
    # 通常用于 MultiStepLR 调度器和 CosineAnnealingLR 调度器。
    scheduler.add_argument(
        "--early_stopping",
        default=-1,
        type=int,
        metavar="N",
        help="early stopping (default: -1). \
            This parameter controls whether to enable early stopping technique. \
            Early stopping is used to terminate training early \
            when the performance on the validation set \
            no longer improves. -1 means early stopping is disabled, \
            other positive integer values mean the specific number of epochs \
            after which to stop training \
            when the performance on the validation set stops improving.",
    )
    # 控制是否启用早期停止
    # 早期停止用于在验证集性能不再提升时提前终止训练。
    # -1 表示禁用早期停止，其他正整数值表示在验证集性能停止提升后的具体纪元数。

    return parser.parse_args()


def load_or_create_config(config: Dict, path_dict: Dict) -> Dict:
    """
    Load an existing config file or create a new one if it doesn't exist.

    Args:
        config (dict): The configuration dictionary.
        path_dict (dict): The dictionary containing paths for model directory.

    Returns:
        dict: The loaded or created configuration dictionary.
    """

    # Load config file with or without Deep Supervision
    if config["name"] is None:
        if config["deep_supervision"]:
            config["name"] = f"{config['dataset']}_{config['arch']}_DS"
        else:
            config["name"] = f"{config['dataset']}_{config['arch']}"

    result_dir = os.path.join(path_dict["result_dir"], config["name"])
    os.makedirs(result_dir, exist_ok=True)

    result_dir = os.path.join(result_dir, "config.yml")
    if os.path.exists(result_dir):
        with open(result_dir, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        logging.info("The configuration file was read successfully.")
    else:
        with open(result_dir, "w", encoding="utf-8") as file:
            yaml.dump(config, file)
        logging.info("The configuration file is successfully written.")

    return config
