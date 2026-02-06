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

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 从blocks包导入通信系统组件 / Import from blocks package
# ==========================================
from blocks.quantizer import UniformQuantizer
from blocks.converter import ADConverter
from blocks.modulator import QPSKModulator
from blocks.channel import AWGNChannel
from blocks.error_correction import Hamming1511
from blocks.utils import (
    calculate_signal_power,
    calculate_uniform_quantization_snr,
    calculate_practical_snr,
    calculate_qpsk_ber,
    calculate_bit_error_rate
)


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
    level_indices = np.clip(level_indices, 0, quantizer.num_levels - 1)
    
    bit_stream = adc_dac.encode(level_indices, bit_depth)
    
    # 3. 信道编码（汉明）
    # Record original length for accurate truncation after decoding
    # 记录原始长度以便解码后精确截断
    original_bit_len = len(bit_stream)
    
    if hamming is not None:
        bit_stream_encoded = hamming.encode(bit_stream)
        # Ensure even length for QPSK modulation (each symbol carries 2 bits)
        # 确保偶数长度以适应QPSK调制（每符号携带2比特）
        if len(bit_stream_encoded) % 2 != 0:
            bit_stream_encoded = np.append(bit_stream_encoded, 0)
    else:
        bit_stream_encoded = bit_stream
        # Ensure even length for QPSK
        if len(bit_stream_encoded) % 2 != 0:
            bit_stream_encoded = np.append(bit_stream_encoded, 0)
    
    # 4. QPSK调制
    modulated_signal = qpsk.modulate(bit_stream_encoded)
    
    # 计算理论QPSK误码率
    # noise_power is linear power, N0 = noise_power / symbol_rate, simplified here
    # noise_power是线性功率，N0 = noise_power / 符号率，此处简化处理
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
    
    # 长度对齐：确保接收比特流与发送编码比特流长度一致
    # Length alignment: ensure received bit stream matches transmitted encoded bit stream length
    min_len = min(len(bit_stream_encoded), len(demodulated_bits))
    tx_aligned = bit_stream_encoded[:min_len]
    rx_aligned = demodulated_bits[:min_len]
    
    # For FEC, ensure rx_aligned length is multiple of 15 to avoid padding issues in decoder
    # 对于FEC情况，确保rx_aligned长度是15的倍数，避免解码器补零干扰最后一个码字
    if hamming is not None:
        remainder = len(rx_aligned) % 15
        if remainder != 0:
            # Truncate to nearest lower multiple of 15
            # 截断至最近的15的倍数（丢弃不完整的最后一个码字）
            valid_len = len(rx_aligned) - remainder
            rx_aligned = rx_aligned[:valid_len]
            tx_aligned = tx_aligned[:valid_len]
    
    # 计算解调后原始误码率（在纠错前）
    ber_before_correction = calculate_bit_error_rate(tx_aligned, rx_aligned)
    
    # 7. 信道解码（汉明）
    # Decode and truncate to original length to remove padding bits added during encoding
    # 解码并截断至原始长度，去除编码时添加的补零位
    ber_after_correction: Optional[float] = None
    decoded_bits = rx_aligned  # Default: no FEC correction applied
    
    if hamming is not None:
        decoded_bits_full = hamming.decode(rx_aligned)
        # Truncate to original data length (remove Hamming padding and QPSK padding if any)
        # 截断至原始数据长度（去除汉明编码补零和QPSK补零）
        decoded_bits = decoded_bits_full[:original_bit_len]
        
        # Calculate BER after correction (compare with original bit stream before encoding)
        # 计算纠错后误码率（与原始信息比特比较）
        min_len_dec = min(len(bit_stream), len(decoded_bits))
        ber_after_correction = calculate_bit_error_rate(
            bit_stream[:min_len_dec], decoded_bits[:min_len_dec]
        )
    
    # 8. 信源解码（比特转索引）
    received_indices = adc_dac.decode(decoded_bits, bit_depth)
    
    # 确保长度匹配（去除可能的填充比特造成的多余索引）
    if len(received_indices) > len(input_signal):
        received_indices = received_indices[:len(input_signal)]
    elif len(received_indices) < len(input_signal):
        # Pad with edge values if too short (should not happen in normal operation)
        # 若过短则边缘填充（正常运行中不应发生）
        received_indices = np.pad(received_indices, 
                                  (0, len(input_signal) - len(received_indices)), 
                                  'edge')
    
    # 9. 反量化重建
    reconstructed_signal = quantizer.dequantize(received_indices)
    
    # 计算端到端SNR
    error_signal = reconstructed_signal - input_signal
    snr_reception = calculate_practical_snr(input_signal, error_signal)
    
    # 获取星座图（使用接收信号）
    # Get constellation from received signal (manual calculation for compatibility)
    # 从接收信号获取星座图（手动计算以保持兼容性）
    length = (len(received_signal) // samples_per_symbol) * samples_per_symbol
    shaped = received_signal[:length].reshape(-1, samples_per_symbol)
    symbols = np.mean(shaped, axis=1)
    constellation = (np.real(symbols), np.imag(symbols))
    
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
    if metrics_fec['ber_after_correction'] is not None:
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
        (3, 8),    # 中噪声，8比特（原始参数）
        (3, 6),    # 高噪声，6比特
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