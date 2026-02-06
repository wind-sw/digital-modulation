"""
Digital Communication System Simulation / 数字通信系统仿真

Complete transmission chain simulation from source to sink.
从信源到信宿的完整传输链路仿真。

Author / 作者: [JW] / [JW]
Date / 日期: 2026-02-06
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Tuple, Dict, Any, List
import os
import math  # 添加math模块用于erfc

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# Communication Blocks Implementation / 通信模块实现
# ==========================================

class UniformQuantizer:
    """均匀量化器 / Uniform Quantizer"""
    
    def __init__(self, bit_depth: int, v_max: float, mode: str = "midrise"):
        self.R = bit_depth
        self.v_max = float(v_max)
        self.mode = mode
        self.L = 2 ** bit_depth  # 量化电平数
        self.delta = (2 * self.v_max) / self.L  # 量化步长
        
    def quantize(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        量化信号
        Returns: (quantized_levels, level_indices)
        """
        # 限制信号范围
        signal_clipped = np.clip(signal, -self.v_max, self.v_max - 1e-10)
        
        if self.mode == "midrise":
            # 中升型：零点为判决边界
            indices = np.floor((signal_clipped + self.v_max) / self.delta).astype(int)
            indices = np.clip(indices, 0, self.L - 1)
            levels = -self.v_max + (indices + 0.5) * self.delta
        else:
            # 中平型
            indices = np.round((signal_clipped + self.v_max) / self.delta - 0.5).astype(int)
            indices = np.clip(indices, 0, self.L - 1)
            levels = -self.v_max + indices * self.delta
            
        return levels, indices
    
    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        """反量化"""
        indices = np.clip(indices, 0, self.L - 1)
        if self.mode == "midrise":
            return -self.v_max + (indices + 0.5) * self.delta
        else:
            return -self.v_max + indices * self.delta


class ADConverter:
    """模数/数模转换器 / A/D D/A Converter"""
    
    @staticmethod
    def encode(values: np.ndarray, bit_depth: int) -> np.ndarray:
        """
        将整数数组编码为比特流（MSB first）
        """
        values = values.astype(int)
        bits = np.unpackbits(values.astype(np.uint8))
        # 如果bit_depth < 8，只保留低位
        if bit_depth < 8:
            mask = np.tile([0]*(8-bit_depth) + [1]*bit_depth, len(values))
            bits = bits[mask.astype(bool)]
        return bits
    
    @staticmethod
    def decode(bit_stream: np.ndarray, bit_depth: int) -> np.ndarray:
        """
        将比特流解码为整数数组
        """
        # 补齐到8的倍数
        padding = (8 - (len(bit_stream) % 8)) % 8
        if padding > 0:
            bit_stream = np.pad(bit_stream, (0, padding), 'constant')
        
        bytes_data = np.packbits(bit_stream.astype(np.uint8))
        return bytes_data


class QPSKModulator:
    """QPSK调制解调器 / QPSK Modulator/Demodulator"""
    
    def __init__(self, samples_per_symbol: int = 8, bit_energy: float = 4.0):
        self.sps = samples_per_symbol
        self.Eb = bit_energy
        # QPSK每个符号2比特，符号能量Es = 2*Eb
        self.Es = 2 * bit_energy
        # 归一化因子，确保平均符号能量为Es
        self.scale = np.sqrt(self.Es / 2)
        
    def modulate(self, bit_stream: np.ndarray) -> np.ndarray:
        """
        QPSK调制：每2比特映射到一个符号
        00 -> (1,1), 01 -> (-1,1), 11 -> (-1,-1), 10 -> (1,-1)
        """
        # 确保偶数个比特
        if len(bit_stream) % 2 != 0:
            bit_stream = np.pad(bit_stream, (0, 1), 'constant')
        
        # 将比特流分组为符号（格雷编码映射）
        bits_i = bit_stream[0::2]  # 奇数位（或偶数位，取决于定义）
        bits_q = bit_stream[1::2]  # 另一位
        
        # 映射到星座点：0 -> +1, 1 -> -1
        i = 1 - 2 * bits_i  # 0->1, 1->-1
        q = 1 - 2 * bits_q
        
        # 上采样并应用脉冲成型（简单矩形脉冲）
        symbols = (i + 1j * q) * self.scale
        
        # 每个符号扩展为sps个采样点
        modulated = np.repeat(symbols, self.sps)
        return modulated
    
    def demodulate(self, received_signal: np.ndarray) -> np.ndarray:
        """
        QPSK解调：匹配滤波（积分 dump）+ 判决
        """
        # 确保长度是sps的整数倍
        length = (len(received_signal) // self.sps) * self.sps
        received_signal = received_signal[:length]
        
        # reshape以便积分
        shaped = received_signal.reshape(-1, self.sps)
        
        # 积分dump（匹配滤波器）
        symbols = np.mean(shaped, axis=1)
        
        # 判决
        i_decision = (np.real(symbols) < 0).astype(int)
        q_decision = (np.imag(symbols) < 0).astype(int)
        
        # 并串转换
        bits = np.zeros(len(symbols) * 2, dtype=int)
        bits[0::2] = i_decision
        bits[1::2] = q_decision
        
        return bits
    
    def get_constellation(self, received_signal: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """获取星座图坐标"""
        if received_signal is not None:
            length = (len(received_signal) // self.sps) * self.sps
            shaped = received_signal[:length].reshape(-1, self.sps)
            symbols = np.mean(shaped, axis=1)
            return np.real(symbols), np.imag(symbols)
        else:
            # 理想星座点
            ideal = np.array([1+1j, -1+1j, -1-1j, 1-1j]) * self.scale
            return np.real(ideal), np.imag(ideal)


class AWGNChannel:
    """AWGN信道 / Additive White Gaussian Noise Channel"""
    
    def simulate(self, signal: np.ndarray, noise_power: float) -> np.ndarray:
        """添加高斯白噪声"""
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
        return signal + noise
    
    def simulate_with_header(self, signal: np.ndarray, noise_power: float, 
                            header_length_symbols: int = 8) -> np.ndarray:
        """
        仿真传输，保护前N个符号（Header）不受噪声影响
        """
        received = self.simulate(signal, noise_power)
        
        # 保护Header：用原始信号替换
        header_samples = header_length_symbols * 8  # 假设每个符号8个采样点
        if header_samples > 0 and header_samples < len(signal):
            received[:header_samples] = signal[:header_samples]
        
        return received


class Hamming1511:
    """
    汉明(15,11)编码器 / Hamming(15,11) Encoder/Decoder
    11个数据比特 + 4个校验比特 = 15个码字符号
    可纠正1位错误
    """
    
    def __init__(self):
        # 生成矩阵 G (11x15)
        self.G = np.array([
            [1,0,0,0,0,0,0,0,0,0,0, 1,1,0,0],
            [0,1,0,0,0,0,0,0,0,0,0, 1,0,1,0],
            [0,0,1,0,0,0,0,0,0,0,0, 0,1,1,0],
            [0,0,0,1,0,0,0,0,0,0,0, 1,1,1,0],
            [0,0,0,0,1,0,0,0,0,0,0, 1,0,0,1],
            [0,0,0,0,0,1,0,0,0,0,0, 0,1,0,1],
            [0,0,0,0,0,0,1,0,0,0,0, 1,1,0,1],
            [0,0,0,0,0,0,0,1,0,0,0, 0,0,1,1],
            [0,0,0,0,0,0,0,0,1,0,0, 1,0,1,1],
            [0,0,0,0,0,0,0,0,0,1,0, 0,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,1, 1,1,1,1]
        ], dtype=int)
        
        # 校验矩阵 H (4x15)
        self.H = np.array([
            [1,1,0,1,1,0,1,0,1,0,1, 1,0,0,0],
            [1,0,1,1,0,1,1,0,0,1,1, 0,1,0,0],
            [0,1,1,1,0,0,0,1,1,1,1, 0,0,1,0],
            [0,0,0,0,1,1,1,1,1,1,1, 0,0,0,1]
        ], dtype=int)
        
        self.n = 15  # 码长
        self.k = 11  # 信息位长
        
    def encode(self, bit_stream: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        编码
        
        Returns:
            Tuple[np.ndarray, int]: (编码后的比特流, padding长度)
        """
        # 补齐到11的倍数
        padding = (self.k - (len(bit_stream) % self.k)) % self.k
        if padding > 0:
            bit_stream = np.pad(bit_stream, (0, padding), 'constant')
        
        # 分组编码
        data = bit_stream.reshape(-1, self.k)
        codewords = np.mod(data @ self.G, 2)
        return codewords.flatten(), padding  # 返回padding信息以便解码时去除
    
    def decode(self, received: np.ndarray, padding: int = 0) -> np.ndarray:
        """
        译码（硬判决+纠错）
        """
        # 补齐到15的倍数
        if len(received) % self.n != 0:
            pad_len = self.n - (len(received) % self.n)
            received = np.pad(received, (0, pad_len), 'constant')
        
        codewords = received.reshape(-1, self.n)
        
        # 计算伴随式
        syndrome = np.mod(codewords @ self.H.T, 2)
        
        # 查找错误图样并纠正
        decoded = []
        for i, (cw, synd) in enumerate(zip(codewords, syndrome)):
            if np.any(synd):
                # 将伴随式转换为整数索引
                err_pos = int(''.join(map(str, synd[::-1])), 2) - 1
                if 0 <= err_pos < self.n:
                    cw = cw.copy()
                    cw[err_pos] = 1 - cw[err_pos]  # 翻转错误比特
            decoded.extend(cw[:self.k])  # 取前11位信息位
        
        result = np.array(decoded)
        # 去除padding
        if padding > 0:
            result = result[:-padding]
        return result


# ==========================================
# Utility Functions / 工具函数
# ==========================================

def calculate_signal_power(signal: np.ndarray) -> float:
    """计算信号功率"""
    return float(np.mean(np.abs(signal)**2))

def calculate_uniform_quantization_snr(R: int, signal_power: float, v_max: float) -> float:
    """
    计算理论量化SNR（dB）
    对于均匀量化：SNR ≈ 6.02*R + 10*log10(12) - 10*log10((2*v_max)^2/signal_power)
    """
    if signal_power <= 0:
        return 0
    # 简化公式：假设信号均匀分布在[-v_max, v_max]
    snr_linear = (3 * (2**R)**2 * signal_power) / ((2*v_max)**2)
    return 10 * np.log10(snr_linear)

def calculate_practical_snr(original: np.ndarray, error: np.ndarray) -> float:
    """计算实际SNR（dB）"""
    signal_power = np.mean(original**2)
    noise_power = np.mean(error**2)
    if noise_power <= 1e-15:
        return 100.0  # 上限
    return 10 * np.log10(signal_power / noise_power)

def calculate_qpsk_ber(Eb: float, N0: float) -> float:
    """
    计算QPSK理论误码率
    QPSK等效于两路正交BPSK，每路误码率Q(sqrt(2*Eb/N0))
    总误码率近似等于单路误码率（因为符号错误主要由单比特错误主导）
    """
    if N0 <= 0:
        return 0.0
    snr = np.sqrt(Eb / N0)
    # 使用math.erfc替代np.erfc（NumPy 2.0+已移除）
    ber = 0.5 * math.erfc(snr)
    return float(ber)

def calculate_bit_error_rate(tx: np.ndarray, rx: np.ndarray) -> float:
    """计算比特误码率"""
    min_len = min(len(tx), len(rx))
    if min_len == 0:
        return 0.0
    errors = np.sum(tx[:min_len] != rx[:min_len])
    return float(errors / min_len)


# ==========================================
# Main Simulation Functions / 主仿真函数
# ==========================================

def simulate_communication_system(input_signal: np.ndarray, bit_depth: int = 8, 
                                  enable_error_correction: bool = False, 
                                  noise_power: float = 0.1) -> Tuple[np.ndarray, tuple, Dict[str, Any]]:
    """
    Digital Communication System Simulation / 数字通信系统仿真
    
    Simulates complete chain: Quantization → Encoding → Modulation → 
    Channel → Demodulation → Decoding → Reconstruction.
    
    仿真完整链路：量化 → 编码 → 调制 → 信道 → 解调 → 解码 → 重建。
    """
    # 参数设置
    v_max = max(255.0, np.max(input_signal))  # 图像像素通常0-255
    samples_per_symbol = 8
    bit_energy = 4.0
    
    # 初始化模块
    quantizer = UniformQuantizer(bit_depth, v_max, "midrise")
    adc_dac = ADConverter()
    qpsk = QPSKModulator(samples_per_symbol, bit_energy)
    channel = AWGNChannel()
    hamming: Optional[Hamming1511] = Hamming1511() if enable_error_correction else None
    
    # ==========================================
    # Transmitter Side / 发送端
    # ==========================================
    
    # 1. 量化
    quantized_signal, quant_levels = quantizer.quantize(input_signal)
    
    # 计算理论量化SNR
    signal_power = calculate_signal_power(input_signal)
    snr_quant_theory = calculate_uniform_quantization_snr(bit_depth, signal_power, v_max)
    
    # 2. 信源编码（电平索引转比特）
    # 注意：UniformQuantizer返回的是电平值，我们需要的是索引
    # 重新计算索引（确保范围0-2^R-1）
    level_indices = ((quantized_signal + v_max) / quantizer.delta - 0.5).astype(int)
    level_indices = np.clip(level_indices, 0, quantizer.L - 1)
    
    bit_stream = adc_dac.encode(level_indices, bit_depth)
    
    # 3. 信道编码（汉明）
    padding_info = 0
    if hamming is not None:
        bit_stream_encoded, padding_info = hamming.encode(bit_stream)
    else:
        bit_stream_encoded = bit_stream
    
    # 4. QPSK调制
    modulated_signal = qpsk.modulate(bit_stream_encoded)
    
    # 计算理论QPSK误码率
    # noise_power是线性功率，N0 = noise_power / 符号率，这里简化处理
    ber_qpsk_theory = calculate_qpsk_ber(bit_energy, noise_power)
    
    # ==========================================
    # Channel / 信道
    # ==========================================
    
    # 5. AWGN信道（保护前8个符号的头部）
    received_signal = channel.simulate_with_header(
        modulated_signal, noise_power, header_length_symbols=8
    )
    
    # 计算信道SNR
    noise_component = received_signal - modulated_signal
    snr_channel = calculate_practical_snr(modulated_signal, noise_component)
    
    # ==========================================
    # Receiver Side / 接收端
    # ==========================================
    
    # 6. QPSK解调
    demodulated_bits = qpsk.demodulate(received_signal)
    
    # 长度对齐
    min_len = min(len(bit_stream_encoded), len(demodulated_bits))
    tx_aligned = bit_stream_encoded[:min_len]
    rx_aligned = demodulated_bits[:min_len]
    
    # 计算解调后原始误码率
    ber_before_correction = calculate_bit_error_rate(tx_aligned, rx_aligned)
    
    # 7. 信道解码（汉明）
    ber_after_correction: Optional[float] = None
    decoded_bits = rx_aligned  # 默认无纠错
    
    if hamming is not None:
        decoded_bits = hamming.decode(rx_aligned, padding_info)
        # 计算纠错后误码率（与原始信息比特比较）
        min_len_dec = min(len(bit_stream), len(decoded_bits))
        ber_after_correction = calculate_bit_error_rate(
            bit_stream[:min_len_dec], decoded_bits[:min_len_dec]
        )
    
    # 8. 信源解码（比特转索引）
    received_indices = adc_dac.decode(decoded_bits, bit_depth)
    
    # 确保长度匹配
    if len(received_indices) > len(input_signal):
        received_indices = received_indices[:len(input_signal)]
    elif len(received_indices) < len(input_signal):
        received_indices = np.pad(received_indices, 
                                  (0, len(input_signal) - len(received_indices)), 
                                  'edge')
    
    # 9. 反量化重建
    reconstructed_signal = quantizer.dequantize(received_indices)
    
    # 计算端到端SNR
    error_signal = reconstructed_signal - input_signal
    snr_reception = calculate_practical_snr(input_signal, error_signal)
    
    # 获取星座图（使用接收信号）
    constellation = qpsk.get_constellation(received_signal)
    
    # 编译指标
    metrics = {
        'ber_qpsk_theory': float(ber_qpsk_theory),
        'ber_before_correction': float(ber_before_correction),
        'ber_after_correction': float(ber_after_correction) if ber_after_correction is not None else None,
        'snr_quantization_theory_db': float(snr_quant_theory),
        'snr_channel_db': float(snr_channel),
        'snr_reception_db': float(snr_reception),
        'original_length': len(input_signal),
        'bit_depth': bit_depth,
        'enable_fec': enable_error_correction
    }
    
    return reconstructed_signal, constellation, metrics


def process_image_transmission(noise_level: float, bit_depth: int, 
                               image_path: str = "test_images/lena_color.tif") -> None:
    """
    Process image transmission with given parameters / 使用指定参数处理图像传输
    
    修正说明：
    1. 统一传入simulate_communication_system的是原始像素数组，而非比特流
    2. 仿真函数返回的是重建的像素数组，无需再进行packbits操作
    3. Header处理改为在仿真函数外部进行，保持接口简洁
    """
    
    print(f"\n{'='*60}")
    print(f"Processing / 处理中: R={bit_depth}, Noise={noise_level}")
    print(f"{'='*60}")
    
    # 加载图像
    try:
        if not os.path.exists(image_path):
            # 创建测试图像（如果文件不存在）
            print(f"Warning / 警告: {image_path} not found, creating test pattern...")
            # 生成标准测试图像（彩色条纹）
            size = 256
            x = np.linspace(0, 255, size)
            y = np.linspace(0, 255, size)
            X, Y = np.meshgrid(x, y)
            R = (X + Y) % 256
            G = (X * 2) % 256
            B = (Y * 2) % 256
            image_array = np.stack([R, G, B], axis=2).astype(np.uint8)
            image_original = Image.fromarray(image_array)
        else:
            image_original = Image.open(image_path)
            image_array = np.array(image_original)
    except Exception as e:
        print(f"Error loading image / 加载图像错误: {e}")
        return
    
    image_shape = image_array.shape
    print(f"Image shape / 图像尺寸: {image_shape}")
    
    # 展平为1D信号（像素值范围0-255）
    pixels_flat = image_array.reshape(-1).astype(float)
    
    # ==========================================
    # 带纠错仿真 (Simulation WITH FEC)
    # ==========================================
    print("Running simulation WITH Hamming(15,11) FEC / 运行带汉明纠错仿真...")
    start_time = time.time()
    
    signal_with_fec, constellation_fec, metrics_fec = simulate_communication_system(
        pixels_flat.copy(), 
        bit_depth=bit_depth, 
        enable_error_correction=True, 
        noise_power=noise_level
    )
    
    time_fec = time.time() - start_time
    print(f"  Completed in / 耗时: {time_fec:.2f} seconds / 秒")
    
    # ==========================================
    # 无纠错仿真 (Simulation WITHOUT FEC)
    # ==========================================
    print("Running simulation WITHOUT FEC / 运行无纠错仿真...")
    start_time = time.time()
    
    signal_without_fec, constellation_no_fec, metrics_no_fec = simulate_communication_system(
        pixels_flat.copy(), 
        bit_depth=bit_depth, 
        enable_error_correction=False, 
        noise_power=noise_level
    )
    
    time_no_fec = time.time() - start_time
    print(f"  Completed in / 耗时: {time_no_fec:.2f} seconds / 秒")
    
    # ==========================================
    # 重建图像 (Image Reconstruction)
    # ==========================================
    # 修正：仿真函数返回的是重建的像素值，直接reshape即可
    
    # 裁剪/映射到有效像素范围[0, 255]
    signal_with_fec = np.clip(signal_with_fec, 0, 255).astype(np.uint8)
    signal_without_fec = np.clip(signal_without_fec, 0, 255).astype(np.uint8)
    
    # Reshape回图像尺寸
    try:
        image_data_fec = signal_with_fec.reshape(image_shape)
        image_data_no_fec = signal_without_fec.reshape(image_shape)
        
        image_fec = Image.fromarray(image_data_fec)
        image_no_fec = Image.fromarray(image_data_no_fec)
        
        # 创建输出目录
        os.makedirs("generated", exist_ok=True)
        
        # 保存图像
        fec_filename = f"generated/image_with_fec_R{bit_depth}_N{noise_level}.png"
        no_fec_filename = f"generated/image_no_fec_R{bit_depth}_N{noise_level}.png"
        
        image_fec.save(fec_filename)
        image_no_fec.save(no_fec_filename)
        
        print(f"  Saved / 已保存: {fec_filename}")
        print(f"  Saved / 已保存: {no_fec_filename}")
        
    except Exception as e:
        print(f"Error reshaping image / 重建图像错误: {e}")
        return
    
    # ==========================================
    # 可视化 (Visualization)
    # ==========================================
    
    # 1. 图像对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Image Transmission: R={bit_depth}, Noise={noise_level}", 
                 fontsize=14, fontweight='bold')
    
    axes[0].imshow(image_original)
    axes[0].set_title("Original / 原始图像")
    axes[0].axis('off')
    
    ber_fec_text = f"BER: {metrics_fec['ber_after_correction']:.2e}" if metrics_fec['ber_after_correction'] else "N/A"
    axes[1].imshow(image_fec)
    axes[1].set_title(f"With FEC / 带纠错\n{ber_fec_text}\nSNR: {metrics_fec['snr_reception_db']:.1f}dB")
    axes[1].axis('off')
    
    axes[2].imshow(image_no_fec)
    axes[2].set_title(f"Without FEC / 无纠错\nBER: {metrics_no_fec['ber_before_correction']:.2e}\nSNR: {metrics_no_fec['snr_reception_db']:.1f}dB")
    axes[2].axis('off')
    
    plt.tight_layout()
    comparison_filename = f"generated/comparison_R{bit_depth}_N{noise_level}.png"
    plt.savefig(comparison_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison / 保存对比图: {comparison_filename}")
    
    # 2. 星座图对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(constellation_fec[0], constellation_fec[1], c='blue', alpha=0.5, s=10)
    axes[0].set_title("Constellation with FEC / 带纠错星座图")
    axes[0].set_xlabel("In-phase / 同相")
    axes[0].set_ylabel("Quadrature / 正交")
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    axes[1].scatter(constellation_no_fec[0], constellation_no_fec[1], c='red', alpha=0.5, s=10)
    axes[1].set_title("Constellation without FEC / 无纠错星座图")
    axes[1].set_xlabel("In-phase / 同相")
    axes[1].set_ylabel("Quadrature / 正交")
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')
    
    plt.tight_layout()
    constellation_filename = f"generated/constellation_R{bit_depth}_N{noise_level}.png"
    plt.savefig(constellation_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved constellation / 保存星座图: {constellation_filename}")
    
    # ==========================================
    # 性能报告 (Performance Report)
    # ==========================================
    print(f"\nMetrics Summary / 性能指标摘要:")
    print(f"  {'Parameter':<35} {'With FEC':>15} {'Without FEC':>15}")
    print(f"  {'-'*35} {'-'*15} {'-'*15}")
    print(f"  {'Quantization SNR (theory)':<35} {metrics_fec['snr_quantization_theory_db']:>15.2f} dB")
    print(f"  {'Channel SNR':<35} {metrics_fec['snr_channel_db']:>15.2f} dB {metrics_no_fec['snr_channel_db']:>15.2f} dB")
    print(f"  {'Reception SNR':<35} {metrics_fec['snr_reception_db']:>15.2f} dB {metrics_no_fec['snr_reception_db']:>15.2f} dB")
    print(f"  {'BER before correction':<35} {metrics_fec['ber_before_correction']:>15.2e} {metrics_no_fec['ber_before_correction']:>15.2e}")
    if metrics_fec['ber_after_correction']:
        print(f"  {'BER after correction':<35} {metrics_fec['ber_after_correction']:>15.2e} {'N/A':>15}")


if __name__ == "__main__":
    # 创建测试目录
    os.makedirs("generated", exist_ok=True)
    os.makedirs("test_images", exist_ok=True)
    
    print("Digital Communication System Simulation / 数字通信系统仿真")
    print("=" * 60)
    print("Complete transmission chain: Source → Quantization → FEC →")
    print("QPSK Modulation → AWGN Channel → Demodulation → Reconstruction")
    print("完整传输链路：信源 → 量化 → 前向纠错 → QPSK调制 →")
    print("AWGN信道 → 解调 → 重建")
    
    # 运行不同参数组合的仿真
    test_cases = [
        (0.05, 8),   # 低噪声，8比特
        (0.1, 8),    # 中噪声，8比特（原始参数）
        (0.2, 6),    # 高噪声，6比特
    ]
    
    for noise, bits in test_cases:
        try:
            process_image_transmission(noise_level=noise, bit_depth=bits)
        except Exception as e:
            print(f"Error in case (noise={noise}, bits={bits}): {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Simulation completed / 仿真完成")
    print("Check 'generated/' folder for results / 请查看generated文件夹中的结果")