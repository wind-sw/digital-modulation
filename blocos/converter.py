"""
Analog-to-Digital / Digital-to-Analog Converter / 模数/数模转换器

Handles conversion between integer quantization levels and binary bit streams.
处理整数量化电平与二进制比特流之间的转换。
"""

import numpy as np


class ADConverter:
    """
    ADC/DAC Converter for int-bit conversion / 整数-比特转换器
    
    This class provides methods to encode integer signal levels into binary 
    representation and decode them back. Supports configurable bit depth.
    
    提供将整数量化电平编码为二进制表示并解码的方法。支持可配置比特深度。
    """
    
    def __init__(self):
        """
        Initialize converter / 初始化转换器
        """
        pass
    
    def encode(self, levels, bit_depth):
        """
        Encode integer levels to binary bit stream / 将整数电平编码为二进制比特流
        
        Converts quantization level indices to binary representation.
        将量化电平索引转换为二进制表示。
        
        Parameters / 参数:
        ----------------
        levels : np.ndarray (int)
            Quantization level indices / 量化电平索引
        bit_depth : int
            Number of bits per sample (R) / 每采样点比特数R
            
        Returns / 返回:
        -------------
        bit_stream : np.ndarray (int, 0/1)
            Flattened binary bit stream / 展平的二进制比特流
        """
        # Ensure integer type / 确保整数类型
        levels = np.asarray(levels, dtype=int)
        
        # Convert to unsigned integer representation / 转换为无符号整数表示
        # Handle negative values if using mid-tread / 中平型量化需处理负值
        if np.any(levels < 0):
            offset = 2 ** (bit_depth - 1)
            levels = levels + offset
        
        # Unpack bits: each integer becomes 'bit_depth' binary digits
        # 解包比特：每个整数展开为'bit_depth'个二进制位
        bit_stream = np.unpackbits(
            levels.astype(np.uint8), 
            bitorder='big'
        )
        
        # If bit_depth < 8, truncate excess bits / 若bit_depth<8，截断多余比特
        if bit_depth < 8:
            mask = np.tile([0] * (8 - bit_depth) + [1] * bit_depth, len(levels))
            bit_stream = bit_stream[mask.astype(bool)]
        
        return bit_stream
    
    def decode(self, bit_stream, bit_depth):
        """
        Decode binary bit stream to integer levels / 将二进制比特流解码为整数电平
        
        Packs binary bits back into integers representing quantization levels.
        将二进制比特打包回表示量化电平的整数。
        
        Parameters / 参数:
        ----------------
        bit_stream : np.ndarray (int, 0/1)
            Binary bit stream / 二进制比特流
        bit_depth : int
            Number of bits per sample / 每采样点比特数
            
        Returns / 返回:
        -------------
        levels : np.ndarray (int)
            Reconstructed quantization level indices / 重建的量化电平索引
        """
        # Ensure bit_stream length is multiple of bit_depth / 确保比特流长度是bit_depth的倍数
        length = len(bit_stream)
        if length % bit_depth != 0:
            # Pad with zeros / 补零对齐
            padding = bit_depth - (length % bit_depth)
            bit_stream = np.concatenate([bit_stream, np.zeros(padding, dtype=int)])
        
        # Reshape to (n_samples, bit_depth) / 重塑为(采样点数, 比特深度)
        bit_matrix = bit_stream.reshape(-1, bit_depth)
        
        # Pack bits to integers / 打包比特为整数
        # Convert to uint8 first (pad with leading zeros to make 8 bits)
        # 先转为uint8（前面补零凑足8位）
        if bit_depth < 8:
            padding = np.zeros((bit_matrix.shape[0], 8 - bit_depth), dtype=np.uint8)
            full_bytes = np.concatenate([padding, bit_matrix.astype(np.uint8)], axis=1)
        else:
            full_bytes = bit_matrix.astype(np.uint8)
        
        levels = np.packbits(full_bytes, axis=1, bitorder='big').flatten()
        
        # Apply offset correction for signed representation / 有符号表示的偏移校正
        if bit_depth > 1:
            offset = 2 ** (bit_depth - 1)
            levels = levels - offset
            
        return levels