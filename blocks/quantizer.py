"""
Uniform Quantization Module / 均匀量化模块

Implements uniform quantization with mid-rise and mid-tread characteristics.
实现中升型(mid-rise)和中平型(mid-tread)两种均匀量化特性。
"""

import numpy as np


class UniformQuantizer:
    """
    Uniform Quantizer / 均匀量化器
    
    Quantizes continuous amplitude signals into discrete levels using uniform 
    step sizes. Supports two types:
    - Mid-rise: No zero level, symmetric around zero
    - Mid-tread: Has zero level, suitable for signals with DC component
    
    使用均匀步长将连续幅度信号量化为离散电平。支持两种类型：
    - 中升型：无零电平，关于零对称
    - 中平型：含零电平，适合含直流分量的信号
    """
    
    def __init__(self, bit_depth, v_max, quantizer_type="midrise"):
        """
        Initialize uniform quantizer / 初始化均匀量化器
        
        Parameters / 参数:
        ----------------
        bit_depth : int
            Number of quantization bits (R). Determines L=2^R levels.
            量化比特数R，决定量化电平数L=2^R。
        v_max : float
            Maximum signal amplitude (clipping level) / 信号最大幅度（限幅电平）
        quantizer_type : str, default="midrise"
            Type of quantizer: "midrise" or "midtread"
            量化器类型："midrise"(中升型) 或 "midtread"(中平型)
        """
        self.bit_depth = bit_depth
        self.v_max = v_max
        self.quantizer_type = quantizer_type.lower()
        
        # Calculate parameters / 计算参数
        self.num_levels = 2 ** bit_depth
        self.delta = (2 * v_max) / self.num_levels  # Step size / 量化步长
        
        if self.quantizer_type == "midtread":
            # Mid-tread: levels are at ..., -3Δ/2, -Δ/2, Δ/2, 3Δ/2, ...
            # 中平型：电平位于..., -3Δ/2, -Δ/2, Δ/2, 3Δ/2, ...
            self.delta = (2 * v_max) / (self.num_levels - 1)
    
    def quantize(self, signal):
        """
        Quantize input signal / 对输入信号进行量化
        
        Parameters / 参数:
        ----------------
        signal : np.ndarray
            Input analog signal / 输入模拟信号
            
        Returns / 返回:
        -------------
        quantized_signal : np.ndarray
            Quantized signal values (actual reconstruction levels)
            量化后的信号值（实际重建电平）
        level_indices : np.ndarray
            Integer indices of quantization levels [0, 2^R-1]
            量化电平的整数索引[0, 2^R-1]
        """
        # 确保是 numpy 数组且为实数
        signal = np.asarray(signal, dtype=np.float64)
    
        # 防止 v_max 为 0
        if self.v_max == 0:
            self.v_max = 1.0


        # Clip signal to [-Vmax, Vmax] / 限幅至[-Vmax, Vmax]
        signal_clipped = np.clip(signal, -self.v_max, self.v_max)
        
        if self.quantizer_type == "midrise":
            # Mid-rise quantization / 中升型量化
            # Shift to positive, scale, and floor / 平移至正数域、缩放、向下取整
            shifted = signal_clipped + self.v_max
            level_indices = np.floor(shifted / self.delta).astype(int)
            # Clip to valid range [0, L-1] / 裁剪至有效范围[0, L-1]
            level_indices = np.clip(level_indices, 0, self.num_levels - 1)
            # Reconstruction level is at midpoint / 重建电平取区间中点
            quantized_signal = level_indices * self.delta + self.delta / 2 - self.v_max
            
        elif self.quantizer_type == "midtread":
            # Mid-tread quantization / 中平型量化
            # Add Vmax, divide by delta, round to nearest integer
            # 加Vmax，除delta，四舍五入
            level_indices = np.round((signal_clipped + self.v_max) / self.delta).astype(int)
            level_indices = np.clip(level_indices, 0, self.num_levels - 1)
            # Reconstruction at exact level / 在精确电平处重建
            quantized_signal = level_indices * self.delta - self.v_max
            
        else:
            raise ValueError(f"Unknown quantizer type: {self.quantizer_type}")
            
        return quantized_signal, level_indices
    
    def dequantize(self, level_indices):
        """
        Reconstruct signal from level indices / 从电平索引重建信号
        
        Parameters / 参数:
        ----------------
        level_indices : np.ndarray (int)
            Quantization level indices / 量化电平索引
            
        Returns / 返回:
        -------------
        reconstructed_signal : np.ndarray
            Reconstructed analog signal / 重建的模拟信号
        """
        level_indices = np.asarray(level_indices, dtype=int)
        
        if self.quantizer_type == "midrise":
            reconstructed = level_indices * self.delta + self.delta / 2 - self.v_max
        else:  # midtread
            reconstructed = level_indices * self.delta - self.v_max
            
        return reconstructed