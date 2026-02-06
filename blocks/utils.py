"""
Utility Functions for Digital Communications / 数字通信工具函数

Mathematical utilities for signal power calculation, SNR/BER computation.
信号功率计算、信噪比/误码率计算的数学工具函数。
"""

import numpy as np


def calculate_signal_power(signal):
    """
    Calculate average signal power / 计算信号平均功率
    
    Formula: P = E[|x|²] = (1/N) * Σ|x[n]|²
    公式：功率 = 信号模值平方的均值
    
    Supports both real and complex signals (e.g., baseband QPSK).
    支持实数和复数信号（如基带QPSK信号）。
    
    Parameters / 参数:
    ----------------
    signal : np.ndarray
        Input signal samples (real or complex) / 输入信号采样值（实数或复数）
        
    Returns / 返回:
    -------------
    power : float
        Average power of the signal / 信号的平均功率
    """
    # Use absolute value squared to handle complex signals correctly
    # For complex: |a+jb|² = a² + b², which equals the signal power
    # 使用模值平方以正确处理复数信号，对于复数：|a+jb|² = a² + b²，即为信号功率
    return np.mean(np.abs(signal) ** 2)


def calculate_uniform_quantization_snr(bit_depth, signal_power, v_max):
    """
    Calculate theoretical SNR for uniform quantization / 计算均匀量化的理论信噪比
    
    For uniform quantizers with sufficient levels (R > 4), the theoretical SNR is:
    SNR = 6.02*R + 1.76 + 10*log10(σ_x²/V_max²) (dB)
    
    对于电平数充足的均匀量化器(R>4)，理论信噪比公式为：
    6.02×比特数 + 1.76 + 10×log10(信号方差/峰值²) (分贝)
    
    Parameters / 参数:
    ----------------
    bit_depth : int
        Number of quantization bits (R) / 量化比特数R
    signal_power : float
        Signal power σ_x² / 信号功率σ_x²
    v_max : float
        Peak signal amplitude / 信号峰值幅度
        
    Returns / 返回:
    -------------
    snr_db : float
        Signal-to-noise ratio in decibels / 信噪比（分贝值）
    """
    # Avoid log of zero / 避免对零取对数
    if signal_power <= 0:
        return -np.inf
    
    snr_db = 6.02 * bit_depth + 1.76 + 10 * np.log10(signal_power / (v_max ** 2))
    return snr_db


def calculate_practical_snr(original_signal, noise_signal):
    """
    Calculate practical SNR from signals / 从信号计算实际信噪比
    
    SNR = 10 * log10(P_signal / P_noise)
    信噪比 = 10×log10(信号功率/噪声功率)
    
    Parameters / 参数:
    ----------------
    original_signal : np.ndarray
        Original clean signal / 原始纯净信号
    noise_signal : np.ndarray
        Noise or error component / 噪声或误差分量
        
    Returns / 返回:
    -------------
    snr_db : float
        Signal-to-noise ratio in decibels / 信噪比（分贝）
    """
    signal_power = calculate_signal_power(original_signal)
    noise_power = calculate_signal_power(noise_signal)
    
    if noise_power <= 0:
        return np.inf
    
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    return snr_db


def calculate_qpsk_ber(bit_energy, noise_psd):
    """
    Calculate theoretical bit error rate for QPSK / 计算QPSK理论误码率
    
    QPSK BER formula: Pb = Q(sqrt(2*Eb/N0))
    where Q is the Q-function (tail probability of Gaussian distribution)
    
    QPSK误码率公式：Pb = Q(√(2×Eb/N0))
    其中Q为Q函数（高斯分布的右尾概率）
    
    Parameters / 参数:
    ----------------
    bit_energy : float
        Energy per bit (Eb) / 每比特能量Eb
    noise_psd : float
        Noise power spectral density (N0) / 噪声功率谱密度N0
        
    Returns / 返回:
    -------------
    ber : float
        Bit error probability / 误码概率
    """
    from scipy.special import erfc
    # Q(x) = 0.5 * erfc(x / sqrt(2))
    # For QPSK: Pb = Q(sqrt(2*Eb/N0))
    snr_per_bit = 2 * bit_energy / noise_psd if noise_psd > 0 else np.inf
    ber = 0.5 * erfc(np.sqrt(snr_per_bit / 2))
    return ber


def calculate_bit_error_rate(tx_bits, rx_bits):
    """
    Calculate practical bit error rate (BER) / 计算实际误码率
    
    BER = (Number of bit errors) / (Total number of bits transmitted)
    误码率 = 错误比特数 / 传输总比特数
    
    Parameters / 参数:
    ----------------
    tx_bits : np.ndarray
        Transmitted bit sequence / 发送比特序列
    rx_bits : np.ndarray
        Received bit sequence / 接收比特序列
        
    Returns / 返回:
    -------------
    ber : float
        Bit error rate [0, 1] / 误码率，范围[0,1]
    """
    # Ensure same length / 确保长度一致
    min_length = min(len(tx_bits), len(rx_bits))
    tx_aligned = tx_bits[:min_length]
    rx_aligned = rx_bits[:min_length]
    
    # Count errors / 统计错误数
    errors = np.sum(tx_aligned != rx_aligned)
    ber = errors / min_length if min_length > 0 else 0.0
    return ber


def calculate_noise_psd(noise_power):
    """
    Calculate noise power spectral density N0 / 计算噪声功率谱密度N0
    
    For AWGN channel with bandwidth B: N0 = P_noise / B
    Here we assume normalized bandwidth (B=1) for simulation.
    
    对于带宽为B的AWGN信道：N0 = 噪声功率/B
    此处仿真假设归一化带宽(B=1)。
    
    Parameters / 参数:
    ----------------
    noise_power : float
        Total noise power / 总噪声功率
        
    Returns / 返回:
    -------------
    n0 : float
        Noise power spectral density / 噪声功率谱密度
    """
    # Normalized single-sided PSD / 归一化单边功率谱密度
    return noise_power