# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Qwen2.5-VL 多模态大语言模型

这个包提供了 Qwen2.5-VL 模型的完整实现，包括：
- 配置类：用于定义模型架构参数
- 模型类：实现多模态理解和文本生成功能
- 处理器类：用于预处理图像、视频和文本输入

Qwen2.5-VL 是一个强大的多模态模型，支持：
- 图像理解和描述
- 视频内容分析
- 视觉问答 (VQA)
- 多轮对话
- 任意分辨率图像处理
- 可变长度视频处理

主要特性：
- 基于 Transformer 架构的视觉编码器
- 3D 旋转位置编码 (3D RoPE)
- 窗口注意力机制
- 多模态特征融合
- 支持多种注意力实现（Flash Attention、SDPA 等）
"""
from typing import TYPE_CHECKING

from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure


if TYPE_CHECKING:
    # 配置模块：定义模型架构和参数
    # - Qwen2_5_VLConfig: 整体模型配置
    # - Qwen2_5_VLVisionConfig: 视觉编码器配置
    # - Qwen2_5_VLTextConfig: 文本模型配置
    from .configuration_qwen2_5_vl import *
    
    # 建模模块：实现模型架构和前向传播
    # - Qwen2_5_VLModel: 核心多模态模型
    # - Qwen2_5_VLForConditionalGeneration: 条件生成模型
    # - Qwen2_5_VLPreTrainedModel: 预训练模型基类
    # - 各种组件类：注意力、MLP、嵌入层等
    from .modeling_qwen2_5_vl import *
    
    # 处理器模块：预处理多模态输入数据
    # - Qwen2_5_VLProcessor: 主要处理器类
    # - 各种参数类：图像、视频、文本处理参数
    from .processing_qwen2_5_vl import *
else:
    import sys

    # 使用懒加载模块以提高导入性能
    # 只有在实际使用时才会加载具体的模块
    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
