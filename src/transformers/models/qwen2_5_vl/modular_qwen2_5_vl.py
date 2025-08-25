# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch Qwen2.5-VL 模型实现。

这个文件包含了 Qwen2.5-VL 多模态大语言模型的核心实现，支持图像和视频的理解。
主要组件包括：
- 视觉编码器：处理图像和视频输入
- 文本编码器：处理文本输入
- 多模态融合：将视觉和文本特征进行融合
- 生成模型：基于融合特征生成文本输出
"""

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig, Qwen2VLTextConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    PatchEmbed,
    PatchMerger,
    Qwen2RMSNorm,
    Qwen2VLCausalLMOutputWithPast,
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
    Qwen2VLModelOutputWithPast,
    Qwen2VLPreTrainedModel,
    TransformersKwargs,
    VisionAttention,
    VisionRotaryEmbedding,
)
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLImagesKwargs, Qwen2VLProcessor

from ...activations import ACT2FN
from ...cache_utils import Cache  # KV Cache 核心类，用于管理键值对缓存

# ============================================================================
# KV Cache 机制说明 (Key-Value Cache Mechanism)
# ============================================================================
# 
# KV Cache 是 Transformer 模型中用于加速序列生成的重要优化技术：
#
# 1. 基本原理:
#    - 在自注意力计算中，每个 token 的 key 和 value 向量在生成过程中保持不变
#    - 通过缓存这些向量，避免在每个生成步骤重复计算所有历史 token 的注意力
#    - 显著减少计算量，从 O(n²) 降低到 O(n)，其中 n 是序列长度
#
# 2. 在 Qwen2.5-VL 中的特殊处理:
#    - 多模态输入: 图像/视频特征在首次前向传播后被融合到文本嵌入中
#    - 3D RoPE: 使用三维旋转位置编码处理时空信息，需要特殊的位置索引计算
#    - rope_deltas 缓存: 预计算的位置编码增量，避免重复计算复杂的多模态位置
#
# 3. 生成流程:
#    - 预填充阶段 (Prefill): 处理完整输入序列，创建初始缓存
#    - 增量生成 (Incremental): 每次只处理新生成的 token，更新缓存
#    - 视觉优化: 非首次生成步骤中移除视觉输入，节省计算资源
#
# 4. 缓存结构:
#    - past_key_values: 存储所有层的 key-value 状态
#    - cache_position: 跟踪当前处理的序列位置
#    - rope_deltas: 缓存的多模态位置编码增量
# ============================================================================
from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...modeling_flash_attention_utils import is_flash_attn_available
from ...modeling_layers import GradientCheckpointingLayer
from ...processing_utils import MultiModalData, ProcessingKwargs, Unpack, VideosKwargs
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import is_torchdynamo_compiling, logging
from ...video_utils import VideoInput


if is_flash_attn_available():
    pass


logger = logging.get_logger(__name__)


class Qwen2_5_VLVisionConfig(PretrainedConfig):
    """
    Qwen2.5-VL 视觉编码器配置类。
    
    用于配置视觉 Transformer 的各种参数，包括网络深度、隐藏层大小、注意力头数等。
    
    Args:
        depth (int): Transformer 层数，默认 32
        hidden_size (int): 隐藏层维度，默认 3584
        hidden_act (str): 激活函数类型，默认 "silu"
        intermediate_size (int): MLP 中间层维度，默认 3420
        num_heads (int): 注意力头数，默认 16
        in_channels (int): 输入图像通道数，默认 3 (RGB)
        patch_size (int): 图像块大小，默认 14x14
        spatial_merge_size (int): 空间合并大小，默认 2x2
        temporal_patch_size (int): 时间维度块大小，默认 2
        tokens_per_second (int): 每秒 token 数，默认 4
        window_size (int): 窗口大小，默认 112
        out_hidden_size (int): 输出隐藏层维度，默认 3584
        fullatt_block_indexes (list): 使用全注意力的层索引，默认 [7, 15, 23, 31]
        initializer_range (float): 参数初始化范围，默认 0.02
    """
    model_type = "qwen2_5_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=32,
        hidden_size=3584,
        hidden_act="silu",
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=4,
        window_size=112,
        out_hidden_size=3584,
        fullatt_block_indexes=[7, 15, 23, 31],
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.out_hidden_size = out_hidden_size
        self.initializer_range = initializer_range


class Qwen2_5_VLTextConfig(Qwen2VLTextConfig):
    """
    Qwen2.5-VL 文本编码器配置类，继承自 Qwen2VL 文本配置。
    
    用于配置 Qwen2.5-VL 模型中的文本编码器部分，包括词汇表大小、隐藏层维度、
    注意力头数、层数等语言模型的核心参数。
    
    继承了 Qwen2VL 的所有文本配置参数，确保与 Qwen2VL 系列模型的兼容性。
    主要用于多模态模型中的文本理解和生成部分。
    
    Attributes:
        model_type (str): 模型类型标识符，用于模型注册和识别
    """
    model_type = "qwen2_5_vl_text"


class Qwen2_5_VLConfig(Qwen2VLConfig):
    """
    Qwen2.5-VL 主配置类，包含视觉和文本两个子配置。
    
    这是 Qwen2.5-VL 多模态模型的顶层配置类，负责管理和协调视觉编码器
    和文本编码器的配置参数。通过子配置的方式，将复杂的多模态模型配置
    分解为独立的视觉和文本配置模块。
    
    主要功能：
    - 统一管理多模态模型的所有配置参数
    - 协调视觉和文本编码器的参数设置
    - 提供模型初始化和序列化的配置接口
    - 支持配置的验证和默认值设置
    
    Attributes:
        model_type (str): 模型类型标识符
        sub_configs (dict): 子配置映射，包含视觉和文本配置类
    """
    model_type = "qwen2_5_vl"
    sub_configs = {"vision_config": Qwen2_5_VLVisionConfig, "text_config": Qwen2_5_VLTextConfig}


class Qwen2_5_VLMLP(nn.Module):
    """
    Qwen2.5-VL 多层感知机 (MLP) 模块。
    
    使用 SwiGLU 激活函数的前馈网络，包含门控机制。
    网络结构：hidden_size -> intermediate_size -> hidden_size
    
    Args:
        config: 模型配置对象
        bias (bool): 是否使用偏置项，默认 False
    """
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size  # 隐藏层维度
        self.intermediate_size = config.intermediate_size  # 中间层维度
        # 门控投影层：hidden_size -> intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        # 上投影层：hidden_size -> intermediate_size  
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        # 下投影层：intermediate_size -> hidden_size
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]  # 激活函数

    def forward(self, hidden_state):
        """
        前向传播。
        
        Args:
            hidden_state: 输入张量，shape: (seq_len, hidden_size)
            
        Returns:
            输出张量，shape: (seq_len, hidden_size)
        """
        # SwiGLU: down_proj(act_fn(gate_proj(x)) * up_proj(x))
        # gate_proj(hidden_state): (seq_len, hidden_size) -> (seq_len, intermediate_size)
        # up_proj(hidden_state): (seq_len, hidden_size) -> (seq_len, intermediate_size)
        # 门控激活后与上投影相乘: (seq_len, intermediate_size)
        # down_proj: (seq_len, intermediate_size) -> (seq_len, hidden_size)
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Qwen2_5_VisionPatchEmbed(PatchEmbed):
    """
    Qwen2.5-VL 视觉块嵌入层，继承自基础 PatchEmbed。
    
    将输入的图像或视频分割成固定大小的块，并将每个块嵌入到高维向量空间中。
    对于视频，还会在时间维度上进行分块处理。
    """
    pass


class Qwen2_5_VisionRotaryEmbedding(VisionRotaryEmbedding):
    """
    Qwen2.5-VL 视觉旋转位置编码，继承自基础 VisionRotaryEmbedding。
    
    为视觉 Transformer 提供旋转位置编码，支持 2D 和 3D 位置信息。
    """
    pass


class Qwen2_5_VLPatchMerger(PatchMerger):
    """
    Qwen2.5-VL 块合并器，继承自基础 PatchMerger。
    
    将多个相邻的视觉块合并成更大的块，减少序列长度，提高计算效率。
    
    Args:
        dim (int): 输出维度
        context_dim (int): 输入上下文维度
        spatial_merge_size (int): 空间合并大小，默认 2
    """
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__(dim, context_dim, spatial_merge_size)
        # 使用 RMS 归一化替代 Layer Normalization
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)


class Qwen2_5_VLVisionAttention(VisionAttention):
    """
    Qwen2.5-VL 视觉注意力模块，继承自基础 VisionAttention。
    
    实现多头自注意力机制，支持旋转位置编码和窗口注意力。
    
    Args:
        config (Qwen2_5_VLVisionConfig): 视觉配置对象
    """
    def __init__(self, config: Qwen2_5_VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size  # 注意力维度


class Qwen2_5_VLVisionBlock(GradientCheckpointingLayer):
    """
    Qwen2.5-VL 视觉 Transformer 块。
    
    包含多头自注意力和前馈网络，使用残差连接和 RMS 归一化。
    支持梯度检查点以节省内存。
    
    Args:
        config: 视觉配置对象
        attn_implementation (str): 注意力实现方式，默认 "sdpa"
    """
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        # 注意力前的归一化层
        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        # MLP 前的归一化层
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        # 多头自注意力模块
        self.attn = Qwen2_5_VLVisionAttention(config=config)
        # 前馈网络模块
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            hidden_states: 输入隐藏状态，shape: (seq_len, hidden_size)
            cu_seqlens: 累积序列长度，用于变长注意力
            rotary_pos_emb: 旋转位置编码（可选）
            position_embeddings: 位置嵌入元组（可选）
            
        Returns:
            输出隐藏状态，shape: (seq_len, hidden_size)
        """
        # 注意力分支：残差连接 + 归一化前置
        # norm1(hidden_states): (seq_len, hidden_size)
        # attn(...): (seq_len, hidden_size)
        # hidden_states + attn(...): (seq_len, hidden_size)
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # MLP 分支：残差连接 + 归一化前置
        # norm2(hidden_states): (seq_len, hidden_size)
        # mlp(...): (seq_len, hidden_size)
        # hidden_states + mlp(...): (seq_len, hidden_size)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5_VLPreTrainedModel(Qwen2VLPreTrainedModel):
    """
    Qwen2.5-VL 预训练模型基类，继承自 Qwen2VLPreTrainedModel。
    
    提供模型初始化、权重加载等基础功能。
    """
    pass


class Qwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VLPreTrainedModel):
    """
    Qwen2.5-VL 视觉 Transformer 预训练模型。
    
    实现完整的视觉编码器，包括：
    - 图像/视频块嵌入
    - 旋转位置编码
    - 多层 Transformer 块（支持全注意力和窗口注意力）
    - 空间块合并
    
    支持任意分辨率图像和变长视频处理。
    
    Args:
        config (Qwen2_5_VLVisionConfig): 视觉配置对象
    """
    config: Qwen2_5_VLVisionConfig
    _no_split_modules = ["Qwen2_5_VLVisionBlock"]  # 不可分割的模块列表

    def __init__(self, config, *inputs, **kwargs) -> None:
        """
        初始化视觉 Transformer 模型。
        
        Args:
            config: 视觉配置对象
            *inputs: 额外的位置参数
            **kwargs: 额外的关键字参数
        """
        super().__init__(config, *inputs, **kwargs)
        
        # 空间合并相关参数
        self.spatial_merge_size = config.spatial_merge_size  # 空间合并大小，通常为2
        self.patch_size = config.patch_size  # 图像块大小
        self.fullatt_block_indexes = config.fullatt_block_indexes  # 使用全注意力的层索引
        self.window_size = config.window_size  # 窗口注意力的窗口大小
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size  # 空间合并单元大小

        # 图像/视频块嵌入层
        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,  # 空间块大小
            temporal_patch_size=config.temporal_patch_size,  # 时间块大小（视频）
            in_channels=config.in_channels,  # 输入通道数（RGB为3）
            embed_dim=config.hidden_size,  # 嵌入维度
        )

        # 旋转位置编码
        head_dim = config.hidden_size // config.num_heads  # 每个注意力头的维度
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)  # RoPE维度为头维度的一半

        # Transformer 块列表
        self.blocks = nn.ModuleList([Qwen2_5_VLVisionBlock(config) for _ in range(config.depth)])
        
        # 块合并器：将多个相邻块合并以减少序列长度
        self.merger = Qwen2_5_VLPatchMerger(
            dim=config.out_hidden_size,  # 输出维度
            context_dim=config.hidden_size,  # 输入上下文维度
            spatial_merge_size=config.spatial_merge_size,  # 空间合并大小
        )
        
        # 梯度检查点标志（用于节省内存）
        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw):
        """
        计算旋转位置编码。
        
        为每个视觉块生成 2D 位置编码，考虑空间合并的影响。
        
        Args:
            grid_thw: 网格的时间、高度、宽度信息，shape: (num_grids, 3)
            
        Returns:
            旋转位置编码，shape: (total_patches, head_dim)
        """
        pos_ids = []
        for t, h, w in grid_thw:
            # 生成高度位置索引
            # hpos_ids: (h, w) -> (h//merge_size, merge_size, w//merge_size, merge_size)
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)  # shape: (h, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )  # shape: (h//merge_size, merge_size, w//merge_size, merge_size)
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)  # 重排列维度
            hpos_ids = hpos_ids.flatten()  # shape: (h*w//merge_size^2,)

            # 生成宽度位置索引
            # wpos_ids: (h, w) -> (h//merge_size, merge_size, w//merge_size, merge_size)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)  # shape: (h, w)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )  # shape: (h//merge_size, merge_size, w//merge_size, merge_size)
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)  # 重排列维度
            wpos_ids = wpos_ids.flatten()  # shape: (h*w//merge_size^2,)
            
            # 堆叠 h 和 w 位置索引，并重复 t 次（时间维度）
            # pos_ids: (h*w//merge_size^2, 2) -> (t*h*w//merge_size^2, 2)
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        
        # 拼接所有网格的位置索引
        pos_ids = torch.cat(pos_ids, dim=0)  # shape: (total_patches, 2)
        
        # 计算最大网格尺寸并生成完整的旋转位置编码
        max_grid_size = grid_thw[:, 1:].max()  # 获取最大的 h 或 w
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)  # shape: (max_size, max_size, head_dim)
        
        # 根据位置索引提取对应的旋转位置编码
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)  # shape: (total_patches, head_dim)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        """
        获取窗口注意力的索引和累积序列长度。
        
        将视觉特征划分为固定大小的窗口，用于窗口注意力机制。
        
        Args:
            grid_thw: 网格的时间、高度、宽度信息，shape: (num_grids, 3)
            
        Returns:
            tuple: (window_index, cu_window_seqlens)
                - window_index: 窗口内的索引映射，shape: (total_valid_tokens,)
                - cu_window_seqlens: 累积窗口序列长度列表
        """
        window_index: list = []
        cu_window_seqlens: list = [0]  # 累积序列长度，从0开始
        window_index_id = 0  # 全局索引偏移量
        
        # 计算合并后的窗口大小
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            # 计算合并后的网格尺寸
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,  # 高度方向合并后的尺寸
                grid_w // self.spatial_merge_size,  # 宽度方向合并后的尺寸
            )
            
            # 创建索引张量，shape: (grid_t, llm_grid_h, llm_grid_w)
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            
            # 计算填充大小，使网格能被窗口大小整除
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            
            # 计算窗口数量
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            
            # 对索引进行填充，填充值为-100（表示无效位置）
            # shape: (grid_t, llm_grid_h+pad_h, llm_grid_w+pad_w)
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            
            # 重塑为窗口结构
            # shape: (grid_t, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            
            # 重排列维度并重塑为窗口格式
            # shape: (grid_t, num_windows_h*num_windows_w, vit_merger_window_size, vit_merger_window_size)
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            
            # 计算每个窗口的有效序列长度（非-100的元素数量）
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)  # shape: (grid_t*num_windows,)
            
            # 展平索引并提取有效索引
            index_padded = index_padded.reshape(-1)  # shape: (total_elements,)
            index_new = index_padded[index_padded != -100]  # 过滤掉填充值
            
            # 添加全局偏移量并保存窗口索引
            window_index.append(index_new + window_index_id)
            
            # 计算累积序列长度（考虑空间合并单元）
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            
            # 更新全局索引偏移量
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
            
        # 拼接所有窗口索引
        window_index = torch.cat(window_index, dim=0)  # shape: (total_valid_tokens,)

        return window_index, cu_window_seqlens

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        视觉 Transformer 前向传播。
        
        Args:
            hidden_states: 输入的视觉特征，shape: (seq_len, hidden_size)
            grid_thw: 网格的时间、高度、宽度信息，shape: (num_images_or_videos, 3)

        Returns:
            处理后的视觉特征，shape: (total_merged_patches, out_hidden_size)
        """
        # 1. 块嵌入：将图像/视频分割成块并嵌入
        # hidden_states: (seq_len, hidden_size) -> (total_patches, hidden_size)
        hidden_states = self.patch_embed(hidden_states)
        
        # 2. 计算旋转位置编码
        # rotary_pos_emb: (total_patches, head_dim)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        
        # 3. 获取窗口索引和累积序列长度
        # window_index: (total_valid_tokens,), cu_window_seqlens: list
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        
        # 4. 转换累积序列长度为张量并去重
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )  # shape: (num_windows+1,)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        # 5. 重排列隐藏状态以适应窗口注意力
        seq_len, _ = hidden_states.size()  # seq_len = total_patches
        # 重塑为空间合并单元格式
        # hidden_states: (seq_len, hidden_size) -> (seq_len//merge_unit, merge_unit, hidden_size)
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        # 根据窗口索引重排列
        # hidden_states: (seq_len//merge_unit, merge_unit, hidden_size) -> (total_valid_tokens//merge_unit, merge_unit, hidden_size)
        hidden_states = hidden_states[window_index, :, :]
        # 重新展平
        # hidden_states: (total_valid_tokens//merge_unit, merge_unit, hidden_size) -> (total_valid_tokens, hidden_size)
        hidden_states = hidden_states.reshape(seq_len, -1)
        
        # 6. 同样重排列旋转位置编码
        # rotary_pos_emb: (total_patches, head_dim) -> (total_patches//merge_unit, merge_unit, head_dim)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        # rotary_pos_emb: (total_patches//merge_unit, merge_unit, head_dim) -> (total_valid_tokens//merge_unit, merge_unit, head_dim)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        # rotary_pos_emb: (total_valid_tokens//merge_unit, merge_unit, head_dim) -> (total_valid_tokens, head_dim)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        
        # 7. 准备位置嵌入（余弦和正弦）
        # emb: (total_valid_tokens, head_dim*2)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        # position_embeddings: (cos_emb, sin_emb), 每个都是 (total_valid_tokens, head_dim*2)
        position_embeddings = (emb.cos(), emb.sin())

        # 8. 计算累积序列长度（用于全注意力层）
        # cu_seqlens: (num_sequences+1,)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # 根据以下因素选择数据类型：
            #  - FA2 要求 cu_seqlens_q 必须是 int32 类型
            #  - torch.onnx.export 要求 cu_seqlens_q 与 grid_thw 具有相同的数据类型
            # 详见 https://github.com/huggingface/transformers/pull/34852
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        # 在开头添加0，形成累积序列长度
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # 9. 通过 Transformer 块进行处理
        for layer_num, blk in enumerate(self.blocks):
            # 根据层索引选择注意力类型
            if layer_num in self.fullatt_block_indexes:
                # 全注意力层：使用全局累积序列长度
                cu_seqlens_now = cu_seqlens
            else:
                # 窗口注意力层：使用窗口累积序列长度
                cu_seqlens_now = cu_window_seqlens

            # 通过 Transformer 块
            # hidden_states: (total_valid_tokens, hidden_size) -> (total_valid_tokens, hidden_size)
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        # 10. 块合并：减少序列长度
        # hidden_states: (total_valid_tokens, hidden_size) -> (total_merged_tokens, out_hidden_size)
        hidden_states = self.merger(hidden_states)
        
        # 11. 恢复原始顺序
        # reverse_indices: (total_merged_tokens,)
        reverse_indices = torch.argsort(window_index)
        # hidden_states: (total_merged_tokens, out_hidden_size) -> (total_merged_patches, out_hidden_size)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


class Qwen2_5_VLModelOutputWithPast(Qwen2VLModelOutputWithPast):
    """Qwen2.5-VL 模型输出类，继承自 Qwen2VL 版本。"""
    pass


class Qwen2_5_VLModel(Qwen2VLModel):
    """
    Qwen2.5-VL 核心模型类。
    
    这是 Qwen2.5-VL 的主要模型类，负责处理多模态输入并生成隐藏状态。
    模型结合了视觉编码器和语言模型，支持图像、视频和文本的联合理解。
    
    主要组件：
    - 视觉编码器 (visual)：处理图像和视频输入
    - 语言模型 (language_model)：处理文本和融合后的多模态特征
    - 多模态融合：将视觉特征嵌入到文本序列中
    - 3D RoPE：支持时间、高度、宽度三维位置编码
    
    模型架构特点：
    - 支持任意分辨率的图像输入
    - 支持变长视频序列处理
    - 使用窗口注意力和全注意力的混合机制
    - 动态视觉 token 数量，根据输入内容自适应
    """
    config: Qwen2_5_VLConfig
    base_model_prefix = ""
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        # 初始化视觉编码器，使用 Qwen2.5-VL 特定的视觉 Transformer
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    ## normalize type, send to device.
                    second_per_grid_t = torch.as_tensor(
                        second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
                    )

                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,  # KV Cache: 缓存的键值对，用于加速序列生成
        # - 在首次前向传播时为 None，模型会创建新的缓存
        # - 在后续生成步骤中包含之前计算的 key 和 value 张量
        # - 避免重复计算已处理 token 的注意力权重，显著提升生成速度
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,  # KV Cache 控制: 是否启用键值对缓存机制
        # - True: 启用缓存，返回 past_key_values 用于后续生成步骤
        # - False: 禁用缓存，每次都重新计算所有注意力权重
        # - None: 使用配置文件中的默认设置 (config.use_cache)
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,  # KV Cache 位置索引: 当前生成 token 在缓存中的位置
        # - shape: (sequence_length,) 表示当前处理的 token 位置
        # - 用于确定在缓存中存储/检索 key-value 的位置
        # - 在增量生成时帮助正确更新缓存内容
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
        """
        Qwen2.5-VL 模型前向传播。
        
        处理多模态输入（文本、图像、视频），将视觉特征与文本嵌入融合，
        然后通过语言模型生成输出。
        
        Args:
            input_ids: 输入的 token IDs，shape: (batch_size, seq_len)
            attention_mask: 注意力掩码，shape: (batch_size, seq_len)
            position_ids: 位置 IDs，shape: (3, batch_size, seq_len) 用于 3D RoPE
            past_key_values: 缓存的键值对
            inputs_embeds: 输入嵌入，shape: (batch_size, seq_len, hidden_size)
            pixel_values: 图像像素值，shape: (num_images, channels, height, width)
            pixel_values_videos: 视频像素值，shape: (num_videos, frames, channels, height, width)
            image_grid_thw: 图像网格的时间、高度、宽度，shape: (num_images, 3)
            video_grid_thw: 视频网格的时间、高度、宽度，shape: (num_videos, 3)
            rope_deltas: RoPE 索引差值，shape: (batch_size,)
            cache_position: 缓存位置，shape: (seq_len,)
            second_per_grid_ts: 每个网格的时间间隔（秒），shape: (num_videos,)
            
        Returns:
            模型输出，包含最后隐藏状态、过去键值对等
        """

        # 1. 设置输出配置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 2. 获取文本嵌入
        if inputs_embeds is None:
            # inputs_embeds: (batch_size, seq_len, hidden_size)
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # 3. 处理图像输入
        if pixel_values is not None:
            # 提取图像特征
            # image_embeds: list of tensors -> (total_image_patches, hidden_size)
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            # 拼接所有图像特征
            # image_embeds: (total_image_patches, hidden_size)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            # 获取图像占位符掩码
            # image_mask: (batch_size, seq_len, hidden_size)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            # 将图像特征嵌入到文本序列中
            # inputs_embeds: (batch_size, seq_len, hidden_size)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # 4. 处理视频输入
        if pixel_values_videos is not None:
            # 提取视频特征
            # video_embeds: list of tensors -> (total_video_patches, hidden_size)
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            # 拼接所有视频特征
            # video_embeds: (total_video_patches, hidden_size)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            # 获取视频占位符掩码
            # video_mask: (batch_size, seq_len, hidden_size)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            # 将视频特征嵌入到文本序列中
            # inputs_embeds: (batch_size, seq_len, hidden_size)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # 5. 计算位置编码
        if position_ids is None:
            # 仅在预填充阶段计算一次 RoPE 索引
            # 编译时无法检查张量值，因此只检查输入长度
            # 可以安全地假设 `length!=1` 意味着我们处于预填充阶段，
            # 因为编译模型目前无法进行辅助解码
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)  # KV Cache: 缓存位置为0，表示预填充阶段开始
                or (past_key_values is None or past_key_values.get_seq_length() == 0)  # KV Cache: 无缓存或缓存为空
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                # 计算 3D RoPE 索引（用于多模态位置编码）
                # position_ids: (3, batch_size, seq_len), rope_deltas: (batch_size, 1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                # KV Cache 优化: 缓存 rope_deltas 用于后续生成步骤
                # 避免在每个生成步骤重复计算 RoPE 位置编码，提升推理效率
                self.rope_deltas = rope_deltas
            else:
                # KV Cache 复用: 后续生成步骤使用缓存的 rope_deltas
                # 这是增量生成的关键优化，避免重复计算位置编码
                batch_size, seq_length, _ = inputs_embeds.shape
                # 创建基础位置 IDs
                # position_ids: (seq_length,) -> (3, batch_size, seq_length)
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                # 应用缓存的位置偏移
                if cache_position is not None:
                    # KV Cache 位置计算: 结合缓存位置和预计算的 RoPE 增量
                    # delta: (batch_size, 1) - 当前生成步骤的位置偏移
                    delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                else:
                    # delta: (batch_size, seq_length)
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                # 重复 delta 以匹配批次大小
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                # 添加位置偏移
                # position_ids: (3, batch_size, seq_length)
                position_ids += delta.to(position_ids.device)

        # 6. 通过语言模型处理融合后的多模态嵌入
        # 使用融合了视觉特征的文本嵌入作为输入
        outputs = self.language_model(
            input_ids=None,  # 不使用原始 token IDs，而是使用嵌入
            position_ids=position_ids,  # 3D 位置编码，shape: (3, batch_size, seq_len)
            attention_mask=attention_mask,  # 注意力掩码，shape: (batch_size, seq_len)
            past_key_values=past_key_values,  # KV Cache: 传递缓存的键值对给语言模型
            # - 首次调用时为 None，语言模型会创建新缓存
            # - 后续调用时包含之前层的 key-value 状态
            inputs_embeds=inputs_embeds,  # 融合后的嵌入，shape: (batch_size, seq_len, hidden_size)
            use_cache=use_cache,  # KV Cache 控制: 指示语言模型是否返回更新后的缓存
            output_attentions=output_attentions,  # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=True,  # 返回字典格式
            cache_position=cache_position,  # KV Cache 位置: 告知语言模型当前处理的序列位置
            **kwargs,
        )

        # 7. 构造输出对象
        # 包装语言模型的输出，添加多模态特定的信息
        output = Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,  # 最后隐藏状态，shape: (batch_size, seq_len, hidden_size)
            past_key_values=outputs.past_key_values,  # KV Cache: 更新后的键值对缓存
            # - 包含所有 Transformer 层的 key-value 状态
            # - 用于下一次生成步骤的增量计算
            # - 每层缓存形状: (batch_size, num_heads, seq_len, head_dim)
            hidden_states=outputs.hidden_states,  # 所有层的隐藏状态（如果请求）
            attentions=outputs.attentions,  # 注意力权重（如果请求）
            rope_deltas=self.rope_deltas,  # RoPE 位置偏移，用于后续生成
        )
        # 根据 return_dict 参数返回字典或元组格式
        return output if return_dict else output.to_tuple()


class Qwen2_5_VLCausalLMOutputWithPast(Qwen2VLCausalLMOutputWithPast):
    """Qwen2.5-VL 因果语言模型输出类，继承自 Qwen2VL 版本。"""
    pass


class Qwen2_5_VLForConditionalGeneration(Qwen2VLForConditionalGeneration):
    """
    Qwen2.5-VL 条件生成模型。
    
    这是用于多模态条件文本生成的主要模型类，支持基于图像和视频的文本生成任务。
    模型在 Qwen2_5_VLModel 的基础上添加了语言建模头，用于生成下一个 token 的概率分布。
    
    主要功能：
    - 多模态理解：处理文本、图像和视频输入
    - 文本生成：基于多模态上下文生成连贯的文本
    - 对话系统：支持多轮对话和指令跟随
    - 视觉问答：回答关于图像或视频内容的问题
    """
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,  # KV Cache: 从上一生成步骤传递的缓存键值对
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,  # KV Cache 位置: 当前生成 token 在序列中的位置索引
        position_ids=None,
        use_cache=True,  # KV Cache 控制: 生成过程中默认启用缓存以提升速度
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        """
        为生成阶段准备模型输入。
        
        处理多模态输入的特殊逻辑，包括 3D RoPE 位置编码的计算和缓存。
        在特定情况下避免将图像输入传递给模型以提高效率。
        
        Args:
            input_ids: 输入 token IDs，shape: (batch_size, seq_len)
            past_key_values: 缓存的键值对（可选）
            attention_mask: 注意力掩码（可选）
            inputs_embeds: 输入嵌入（可选）
            cache_position: 缓存位置信息
            position_ids: 位置 IDs（可选）
            use_cache: 是否使用缓存
            pixel_values: 图像像素值（可选）
            pixel_values_videos: 视频像素值（可选）
            image_grid_thw: 图像网格 THW 信息（可选）
            video_grid_thw: 视频网格 THW 信息（可选）
            second_per_grid_ts: 每个网格的时间戳（可选）
            **kwargs: 其他关键字参数
            
        Returns:
            dict: 准备好的模型输入字典
        """
        # 重写父类方法 -- 在特定情况下我们不希望将图像输入传递给模型

        # 调用父类方法获取基础模型输入
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen2.5-VL 的 position_ids 需要结合 rope_deltas 来准备
        if position_ids is None:
            # KV Cache 阶段判断: 仅在预填充阶段计算一次 RoPE 索引
            # 编译时无法检查张量值，因此只检查输入长度
            # 可以安全地假设 `length!=1` 意味着我们处于预填充阶段，
            # 因为编译模型目前无法进行辅助解码
            if cache_position[0] == 0 or self.model.rope_deltas is None:  # KV Cache: 首次生成或无缓存状态
                # 计算视觉位置和 RoPE 增量
                # vision_positions: (3, seq_len, 1), rope_deltas: (seq_len,)
                vision_positions, rope_deltas = self.model.get_rope_index(
                    model_inputs.get("input_ids", None),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                # KV Cache 优化: 缓存 rope_deltas 以供后续生成步骤使用
                # 避免在每个 token 生成时重复计算复杂的多模态位置编码
                self.model.rope_deltas = rope_deltas
            # KV Cache 复用: 使用之前预计算的 rope_deltas 获取正确的位置 IDs
            # 这是增量生成的核心优化，避免重复计算位置编码
            elif "position_ids" in model_inputs:
                # position_ids: (1, seq_len)
                position_ids = model_inputs["position_ids"][None, ...]
                delta = self.model.rope_deltas  # KV Cache: 从缓存中获取预计算的 RoPE 增量, shape: (cached_seq_len,)
                # 重复插值以匹配当前序列长度
                delta = delta.repeat_interleave(position_ids.shape[1] // delta.shape[0], dim=0)
                # 应用 RoPE 增量
                # vision_positions: (1, seq_len, 1) -> (3, seq_len, 1)
                vision_positions = position_ids + delta.expand_as(position_ids)
                vision_positions = vision_positions.expand(3, vision_positions.shape[1], -1)

            # 将 "文本 + 视觉" 位置拼接成 [4, batch_size, seq_len] 的格式
            if "position_ids" not in model_inputs:
                # 为文本生成位置索引
                # text_positions: (1, 1, seq_len)
                text_positions = torch.arange(input_ids, device=input_ids.device)[None, None, :]
            else:
                # 使用现有的位置 IDs
                # text_positions: (1, seq_len) -> (1, 1, seq_len)
                text_positions = model_inputs["position_ids"][None, ...]
            
            # 拼接文本和视觉位置编码
            # model_inputs["position_ids"]: (4, batch_size, seq_len)
            # 其中 4 个维度分别对应：文本位置、视觉 x 位置、视觉 y 位置、视觉 t 位置
            model_inputs["position_ids"] = torch.cat([text_positions, vision_positions], dim=0)

        # KV Cache 优化: 在非首次生成步骤中，移除像素值以节省计算
        # 因为视觉特征已经在首次前向传播中计算并融合到文本嵌入中
        # 后续生成步骤只需要处理新的文本 token，无需重复处理视觉输入
        if cache_position[0] != 0:  # KV Cache: 非首次生成步骤
            model_inputs["pixel_values"] = None  # 清除图像像素值，节省显存和计算
            model_inputs["pixel_values_videos"] = None  # 清除视频像素值，节省显存和计算

        return model_inputs


class Qwen2_5_VLVideosProcessorKwargs(VideosKwargs, total=False):
    """
    Qwen2.5-VL 视频处理器参数类。
    
    定义视频处理的相关参数，包括帧率等视频特定的配置。
    """
    fps: Union[list[float], float]  # 帧率：每秒帧数


class Qwen2_5_VLImagesKwargs(Qwen2VLImagesKwargs):
    """
    Qwen2.5-VL 图像处理器参数类，继承自 Qwen2VL 图像参数。
    
    定义图像处理的相关参数，如尺寸调整、归一化等。
    """
    pass


class Qwen2_5_VLProcessorKwargs(ProcessingKwargs, total=False):
    """
    Qwen2.5-VL 处理器参数类。
    
    整合图像、视频和文本处理的所有参数配置。
    """
    images_kwargs: Qwen2_5_VLImagesKwargs  # 图像处理参数
    videos_kwargs: Qwen2_5_VLVideosProcessorKwargs  # 视频处理参数
    _defaults = {
        "text_kwargs": {
            "padding": False,  # 默认不填充
            "return_mm_token_type_ids": False,  # 默认不返回多模态 token 类型 IDs
        },
    }


class Qwen2_5_VLProcessor(Qwen2VLProcessor):
    """
    Qwen2.5-VL 多模态处理器。
    
    将 Qwen2.5-VL 图像处理器、视频处理器和 Qwen2 分词器封装成单一处理器。
    提供 Qwen2VLImageProcessor、Qwen2_5_VLVideoProcessor 和 Qwen2TokenizerFast 的所有功能。
    
    主要功能：
    - 统一的多模态输入处理接口
    - 图像预处理：调整尺寸、归一化、分块等
    - 视频预处理：帧采样、时序处理等
    - 文本处理：分词、编码、特殊 token 处理
    - 多模态融合：协调不同模态的输入格式
    
    Args:
        image_processor: Qwen2VL 图像处理器（必需）
        tokenizer: Qwen2 分词器（必需）
        video_processor: Qwen2.5-VL 视频处理器（必需）
        chat_template: 用于将对话消息列表转换为可分词字符串的 Jinja 模板（可选）
    
    使用示例：
        ```python
        processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        
        # 处理图像和文本
        inputs = processor(
            text="描述这张图片",
            images=image,
            return_tensors="pt"
        )
        
        # 处理视频和文本
        inputs = processor(
            text="这个视频在做什么？",
            videos=video,
            return_tensors="pt"
        )
        ```
    """

    image_processor_class = "AutoImageProcessor"

    @property
    def model_input_names(self):
        """
        获取模型输入参数名称列表。
        
        合并分词器和图像处理器的输入参数名称，并添加视频处理特有的参数。
        
        Returns:
            list[str]: 模型接受的所有输入参数名称列表，包括：
                - 分词器参数：如 input_ids, attention_mask 等
                - 图像处理器参数：如 pixel_values, image_grid_thw 等
                - 视频特有参数：second_per_grid_ts（每网格秒数）
        """
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        names_from_processor = list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
        return names_from_processor + ["second_per_grid_ts"]

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Qwen2_5_VLProcessorKwargs],
    ) -> BatchFeature:
        """
        为模型准备一个或多个序列和图像/视频的主要方法。
        
        该方法将 `text` 和 `kwargs` 参数转发给 Qwen2TokenizerFast 的 [`~Qwen2TokenizerFast.__call__`] 方法
        来编码文本（如果 `text` 不为 `None`）。为了准备视觉输入，该方法将 `images`/`videos` 和 `kwargs` 
        参数转发给相应的图像/视频处理器。

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                要准备的图像或图像批次。每个图像可以是 PIL 图像、NumPy 数组或 PyTorch 张量。
                支持通道优先和通道最后两种格式。
            text (`str`, `list[str]`, `list[list[str]]`):
                要编码的序列或序列批次。每个序列可以是字符串或字符串列表（预分词字符串）。
                如果序列以字符串列表形式提供（预分词），必须设置 `is_split_into_words=True`。
            videos (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                要准备的视频或视频批次。每个视频可以是 4D NumPy 数组或 PyTorch 张量，
                或者是 3D 帧的嵌套列表。支持通道优先和通道最后两种格式。
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                如果设置，将返回特定框架的张量。可接受的值有：
                - `'tf'`: 返回 TensorFlow `tf.constant` 对象
                - `'pt'`: 返回 PyTorch `torch.Tensor` 对象
                - `'np'`: 返回 NumPy `np.ndarray` 对象
                - `'jax'`: 返回 JAX `jnp.ndarray` 对象

        Returns:
            [`BatchFeature`]: 包含以下字段的 [`BatchFeature`] 对象：

            - **input_ids** -- 要输入模型的 token ID 列表。当 `text` 不为 `None` 时返回。
            - **attention_mask** -- 指定模型应关注哪些 token 的索引列表（当 `return_attention_mask=True` 
              或 "attention_mask" 在 `self.model_input_names` 中且 `text` 不为 `None` 时）。
            - **pixel_values** -- 要输入模型的图像像素值。当 `images` 不为 `None` 时返回。
            - **pixel_values_videos** -- 要输入模型的视频像素值。当 `videos` 不为 `None` 时返回。
            - **image_grid_thw** -- LLM 中图像的 3D 网格列表。当 `images` 不为 `None` 时返回。
            - **video_grid_thw** -- LLM 中视频的 3D 网格列表。当 `videos` 不为 `None` 时返回。
            - **second_per_grid_ts** -- 每个时间网格的视频秒数列表。当 `videos` 不为 `None` 时返回。
        """
        # 1. 合并处理参数
        # 将用户提供的参数与默认参数合并
        output_kwargs = self._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # 2. 初始化图像和视频输入字典
        image_inputs = videos_inputs = {}
        
        # 3. 处理图像输入
        if images is not None:
            # 使用图像处理器处理图像
            # image_inputs: {"pixel_values": tensor, "image_grid_thw": list}
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]  # 图像的 3D 网格信息

        # 4. 处理视频输入
        if videos is not None:
            # 获取帧率参数，默认为 2.0 FPS
            fps = output_kwargs["videos_kwargs"].get("fps", 2.0)
            # 使用视频处理器处理视频
            # videos_inputs: {"pixel_values_videos": tensor, "video_grid_thw": list}
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]  # 视频的 3D 网格信息

            # 计算每个时间网格的秒数
            if isinstance(fps, (int, float)):
                # 单一帧率：为所有视频使用相同的帧率
                second_per_grid_ts = [self.video_processor.temporal_patch_size / fps] * len(video_grid_thw)
            elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                # 多个帧率：每个视频使用不同的帧率
                second_per_grid_ts = [self.video_processor.temporal_patch_size / tmp for tmp in fps]
            else:
                # 帧率参数错误
                raise ValueError(
                    f"fps 的长度 ({len(fps) if hasattr(fps, '__len__') else fps}) 必须等于 video_grid_thw 的长度 ({len(video_grid_thw)})，或者 fps 应该是单个数字。"
                )
            # 添加时间戳信息到视频输入
            videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})

        # 5. 预处理文本输入
        if not isinstance(text, list):
            text = [text]  # 确保文本是列表格式

        text = text.copy()  # 复制文本以避免原地修改
        
        # 6. 处理图像 token 替换
        if images is not None:
            merge_length = self.image_processor.merge_size**2  # 合并长度（通常为4）
            index = 0  # 图像索引
            for i in range(len(text)):
                # 为每个图像 token 计算实际需要的 token 数量
                while self.image_token in text[i]:
                    # 计算当前图像需要的 token 数量
                    # num_image_tokens = (t * h * w) // merge_length
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    # 用占位符替换图像 token
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                # 将占位符替换回图像 token
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        # 7. 处理视频 token 替换
        if videos is not None:
            merge_length = self.video_processor.merge_size**2  # 合并长度（通常为4）
            index = 0  # 视频索引
            for i in range(len(text)):
                # 为每个视频 token 计算实际需要的 token 数量
                while self.video_token in text[i]:
                    # 计算当前视频需要的 token 数量
                    # num_video_tokens = (t * h * w) // merge_length
                    num_video_tokens = video_grid_thw[index].prod() // merge_length
                    # 用占位符替换视频 token
                    text[i] = text[i].replace(self.video_token, "<|placeholder|>" * num_video_tokens, 1)
                    index += 1
                # 将占位符替换回视频 token
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        # 8. 处理文本分词
        # 提取返回张量类型和多模态 token 类型 ID 标志
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        
        # 使用分词器处理文本
        # text_inputs: {"input_ids": list, "attention_mask": list, ...}
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        
        # 检查特殊多模态 token 的有效性
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        # 9. 生成多模态 token 类型 ID（如果需要）
        if return_mm_token_type_ids:
            # 创建多模态 token 类型 ID 数组
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])  # 初始化为0（文本）
            mm_token_type_ids[array_ids == self.image_token_id] = 1  # 图像 token 标记为1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        # 10. 返回合并的批次特征
        # 将文本、图像和视频输入合并为单个 BatchFeature 对象
        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """
        计算给定尺寸的多模态输入所需的占位符 token 数量。
        
        此方法用于预先计算图像和视频输入在模型中需要多少个 token 位置，
        这对于文本生成、内存分配和序列长度规划非常重要。
        
        Args:
            image_sizes (list[list[int]], optional): 图像尺寸列表，每个图像格式为 (height, width)
            video_sizes (list[list[int]], optional): 视频尺寸列表，每个视频格式为 (num_frames, height, width)
            **kwargs: 其他处理参数，如 merge_size 等
        
        Returns:
            MultiModalData: 包含每种输入模态的 token 数量及其他有用数据的对象
                - num_image_tokens: 每张图像的 token 数量列表
                - num_image_patches: 每张图像的块数量列表
                - num_video_tokens: 每个视频的 token 数量列表（如果有视频输入）
        
        计算逻辑:
            1. 图像：patches = (height // patch_size) * (width // patch_size)
                    tokens = patches // (merge_size^2)
            2. 视频：patches = frames * (height // patch_size) * (width // patch_size)
                    tokens = patches // (merge_size^2)
        """

        # 初始化视觉数据字典
        vision_data = {}
        
        # 处理图像输入
        if image_sizes is not None:
            # 获取图像处理参数，合并默认参数和用户提供的参数
            images_kwargs = Qwen2_5_VLProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)
            
            # 获取合并大小参数（用于计算最终 token 数量）
            merge_size = images_kwargs.get("merge_size", None) or self.image_processor.merge_size

            # 计算每张图像的块数量
            # 每张图像被分割成多个 patch，数量取决于图像尺寸和 patch 大小
            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            
            # 计算每张图像的最终 token 数量
            # 通过空间合并减少 token 数量：patches // (merge_size^2)
            num_image_tokens = [(num_patches // merge_size**2) for num_patches in num_image_patches]
            
            # 更新视觉数据字典
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        # 处理视频输入
        if video_sizes is not None:
            # 获取视频处理参数，合并默认参数和用户提供的参数
            videos_kwargs = Qwen2_5_VLProcessorKwargs._defaults.get("videos_kwargs", {})
            videos_kwargs.update(kwargs)
            
            # 计算每个视频的块数量
            # 视频块数量 = 帧数 * 每帧的块数量
            num_video_patches = [
                self.video_processor.get_number_of_video_patches(*video_size, videos_kwargs)
                for video_size in video_sizes
            ]
            
            # 计算每个视频的最终 token 数量
            # 通过空间合并减少 token 数量：patches // (merge_size^2)
            num_video_tokens = [(num_patches // merge_size**2) for num_patches in num_video_patches]
            
            # 添加视频 token 数量到视觉数据字典
            vision_data["num_video_tokens"] = num_video_tokens

        # 返回包含所有多模态数据的对象
        return MultiModalData(**vision_data)


__all__ = [
    "Qwen2_5_VLConfig",
    "Qwen2_5_VLTextConfig",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2_5_VLModel",
    "Qwen2_5_VLPreTrainedModel",
    "Qwen2_5_VLProcessor",
    "Qwen2_5_VLTextModel",  # noqa: F822
]
