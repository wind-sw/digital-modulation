"""
AWGN Channel Simulation / 加性高斯白噪声信道仿真

Simulates the effects of Additive White Gaussian Noise (AWGN) on transmitted signals.
仿真加性高斯白噪声(AWGN)对传输信号的影响。
"""

import numpy as np


class AWGNChannel:
    """
    Additive White Gaussian Noise Channel / 加性高斯白噪声信道
    
    Models the physical channel where transmitted signals are corrupted by 
    thermal noise modeled as Gaussian distributed random process.
    
    对物理信道建模，传输信号被建模为高斯分布随机过程的热噪声所污染。
    """
    
    def __init__(self):
        """
        Initialize AWGN channel / 初始化AWGN信道
        """
        pass
    
    def simulate(self, input_signal, noise_power):
        """
        Simulate channel with AWGN / 仿真带AWGN的信道
        
        Adds complex Gaussian noise with specified power.
        添加指定功率的复数高斯噪声。
        
        Parameters / 参数:
        ----------------
        input_signal : np.ndarray (complex or real)
            Transmitted signal / 发送信号
        noise_power : float
            Noise power (variance σ²) / 噪声功率（方差σ²）
            
        Returns / 返回:
        -------------
        received_signal : np.ndarray
            Signal corrupted by noise / 被噪声污染的信号
        """
        if noise_power <= 0:
            return input_signal.copy()
        
        # Generate complex noise for complex signals, real for real signals
        # 复数信号生成复噪声，实信号生成实噪声
        if np.iscomplexobj(input_signal):
            # Complex noise: I and Q are independent Gaussian
            # 复噪声：I、Q为独立高斯分布
            noise_real = np.random.normal(0, np.sqrt(noise_power/2), len(input_signal))
            noise_imag = np.random.normal(0, np.sqrt(noise_power/2), len(input_signal))
            noise = noise_real + 1j * noise_imag
        else:
            noise = np.random.normal(0, np.sqrt(noise_power), len(input_signal))
        
        received_signal = input_signal + noise
        return received_signal
    
    def simulate_with_header(self, input_signal, noise_power, header_length_symbols):
        """
        Simulate channel protecting header symbols / 仿真信道并保护头部符号
        
        First 'header_length_symbols' symbols pass through without noise
        (simulating perfect header synchronization/pilot).
        
        前'header_length_symbols'个符号无噪声通过（仿真理想的头部同步/导频）。
        
        Parameters / 参数:
        ----------------
        input_signal : np.ndarray
            Transmitted signal / 发送信号
        noise_power : float
            Noise power / 噪声功率
        header_length_symbols : int
            Number of header symbols to protect / 需保护的头部符号数
            
        Returns / 返回:
        -------------
        received_signal : np.ndarray
            Output with clean header and noisy payload / 头部洁净、载荷含噪声的输出
        """
        received = self.simulate(input_signal, noise_power)
        
        # Calculate samples in header / 计算头部采样点数
        # Assuming input_signal is oversampled / 假设input_signal已过采样
        samples_per_symbol = 8  # Default assumption / 默认假设
        header_samples = header_length_symbols * samples_per_symbol
        
        if header_samples < len(input_signal):
            # Keep header clean, rest is already noisy
            # 保持头部洁净，其余部分已有噪声
            received[:header_samples] = input_signal[:header_samples]
            
        return received