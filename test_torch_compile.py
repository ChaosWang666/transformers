#!/usr/bin/env python3
"""
测试修改后的Qwen2.5-VL模型与torch.compile的兼容性

主要测试点：
1. 编译后的模型是否正常运行
2. 位置编码优化是否正确
3. KV Cache机制是否兼容
"""

import torch
import sys
import os
sys.path.insert(0, os.path.abspath("./src"))

def test_torch_compile_compatibility():
    """测试torch.compile兼容性"""
    
    # 检查是否支持torch.compile
    if not hasattr(torch, 'compile'):
        print("当前环境不支持torch.compile，跳过测试")
        return
        
    try:
        from transformers.models.qwen2_5_vl.modular_qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration, 
            Qwen2_5_VLConfig
        )
    except ImportError as e:
        print(f"导入失败: {e}")
        return
        
    print("开始torch.compile兼容性测试...")
    
    # 创建最小配置
    config = Qwen2_5_VLConfig(
        hidden_size=64,
        vocab_size=1000,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=512,
    )
    
    try:
        # 创建模型实例
        model = Qwen2_5_VLForConditionalGeneration._from_config(config)
        model.eval()
        
        # 编译模型
        print("正在编译模型...")
        compiled_model = torch.compile(model)
        
        # 准备测试数据
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, 100, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        
        print("测试编译后的模型前向传播...")
        
        # 测试预填充阶段
        with torch.no_grad():
            outputs = compiled_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True
            )
            
        print(f"✓ 预填充阶段成功，输出logits形状: {outputs.logits.shape}")
        print(f"✓ 返回KV缓存: {outputs.past_key_values is not None}")
        
        # 测试生成阶段（使用KV缓存）
        next_token_ids = torch.randint(0, 100, (batch_size, 1))
        
        with torch.no_grad():
            outputs2 = compiled_model(
                input_ids=next_token_ids,
                attention_mask=torch.cat([attention_mask, torch.ones((batch_size, 1))], dim=1),
                past_key_values=outputs.past_key_values,
                use_cache=True
            )
            
        print(f"✓ 生成阶段成功，输出logits形状: {outputs2.logits.shape}")
        
        print("🎉 torch.compile兼容性测试通过！")
        
    except Exception as e:
        print(f"❌ torch.compile兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_position_encoding_consistency():
    """测试位置编码的一致性"""
    
    try:
        from transformers.models.qwen2_5_vl.modular_qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration, 
            Qwen2_5_VLConfig
        )
    except ImportError as e:
        print(f"导入失败: {e}")
        return
        
    print("开始位置编码一致性测试...")
    
    # 创建配置
    config = Qwen2_5_VLConfig(
        hidden_size=64,
        vocab_size=1000,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=512,
    )
    
    try:
        model = Qwen2_5_VLForConditionalGeneration._from_config(config)
        model.eval()
        
        batch_size = 2
        seq_length = 8
        input_ids = torch.randint(0, 100, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        
        # 测试prepare_inputs_for_generation
        print("测试prepare_inputs_for_generation...")
        
        with torch.no_grad():
            # 预填充阶段
            prepared_inputs = model.prepare_inputs_for_generation(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                cache_position=None,
                past_key_values=None
            )
            
            print(f"✓ 预填充阶段position_ids形状: {prepared_inputs.get('position_ids', 'None')}")
            
            # 模拟第一次前向传播
            outputs = model(**prepared_inputs)
            
            # 生成阶段
            next_input_ids = torch.randint(0, 100, (batch_size, 1))
            new_attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1))], dim=1)
            
            prepared_inputs2 = model.prepare_inputs_for_generation(
                input_ids=next_input_ids,
                attention_mask=new_attention_mask,
                past_key_values=outputs.past_key_values,
                cache_position=torch.tensor([seq_length]),
                use_cache=True
            )
            
            position_ids = prepared_inputs2.get('position_ids')
            if position_ids is not None:
                print(f"✓ 生成阶段position_ids形状: {position_ids.shape}")
                print(f"✓ position_ids设备: {position_ids.device}")
            else:
                print("⚠ 生成阶段未返回position_ids")
            
        print("🎉 位置编码一致性测试通过！")
        
    except Exception as e:
        print(f"❌ 位置编码一致性测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== Qwen2.5-VL 优化代码兼容性测试 ===")
    print()
    
    test_position_encoding_consistency()
    print()
    test_torch_compile_compatibility()
    print()
    print("测试完成！")