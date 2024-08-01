# -*- coding: utf-8 -*-
"""UNet and NestedUNet implementation"""

import torchvision.models as models
from torchinfo import summary

from contextlib import redirect_stdout
from path_config import path_dict

resnet18 = models.resnet18()

# 打开一个文件进行写入
with open('/logs/resnet18_summary.md', 'w', encoding="utf-8") as f:
    # 重定向stdout到文件
    with redirect_stdout(f):
        # 打印模型的summary
        summary(resnet18, (1, 3, 224, 224))  # 3: 图片的通道数, 224: 图片的高宽
