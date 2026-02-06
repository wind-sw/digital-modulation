"""
QPSK Modulation/Demodulation Module / QPSK调制解调模块

Implements Quadrature Phase Shift Keying (QPSK) modulation with pulse shaping.
实现带脉冲成形的正交相移键控(QPSK)调制。
"""

import numpy as np


class QPSKModulator:
    """
    QPSK Modulator and Demodulator / QPSK调制解调器
    
    QPSK maps 2 bits to 4 phase states (π/4, 3π/4, 5π/4, 7π/4).
    Each symbol carries 2 bits of information.
    
    QPSK将2比特映射到4个相位状态(π/4, 3π/4, 5π/4, 7π/4)。
    每个符号携带2比特信息。
    
    Constellation mapping (Gray coding) / 星座映射(格雷码):
    00 → π/4 (I=+1, Q=+1)
    01 → 3π/4 (I=-1, Q=+1)  
    11 → 5π/4 (I=-1, Q=-1)
    10 → 7π/4 (I=+1, Q=-1)
    """
    
    def __init__(self, samples_per_symbol: int = 8, bit_energy: float = 4.0):
        """
        Initialize QPSK modulator / 初始化QPSK调制器
        
        Parameters / 参数:
        ----------------
        samples_per_symbol : int, default=8
            Number of samples per symbol (pulse shaping oversampling)
            每符号采样点数（脉冲成形过采样率）
        bit_energy : float, default=4.0
            Energy per bit (Eb) / 每比特能量Eb (可为浮点数)
        """
        self.samples_per_symbol = samples_per_symbol
        self.bit_energy = float(bit_energy)  # 确保是 float 类型
        
        # Symbol energy (2 bits per symbol) / 符号能量（每符号2比特）
        self.symbol_energy = 2 * self.bit_energy
        # Amplitude scaling factor / 幅度缩放因子
        self.amplitude = np.sqrt(self.symbol_energy / 2)
    
    def modulate(self, bit_stream: np.ndarray) -> np.ndarray:
        """
        Modulate bit stream to QPSK signal / 将比特流调制为QPSK信号
        
        Parameters / 参数:
        ----------------
        bit_stream : np.ndarray (int, 0/1)
            Input binary data stream / 输入二进制数据流
            
        Returns / 返回:
        -------------
        modulated_signal : np.ndarray (complex)
            Complex baseband QPSK signal / 复数基带QPSK信号
        """
        # Ensure even number of bits / 确保比特数为偶数
        if len(bit_stream) % 2 != 0:
            bit_stream = np.append(bit_stream, 0)
        
        # Reshape to (n_symbols, 2) / 重塑为(符号数, 2)
        bit_pairs = bit_stream.reshape(-1, 2)
        
        # Gray coding to phase mapping / 格雷码到相位映射
        # I (in-phase) and Q (quadrature) components
        # I（同相）和Q（正交）分量
        i_component = np.where(bit_pairs[:, 0] == 0, 1, -1)  # MSB / 最高有效位
        q_component = np.where(bit_pairs[:, 1] == 0, 1, -1)  # LSB / 最低有效位
        
        # Create complex symbols / 创建复数符号
        symbols = self.amplitude * (i_component + 1j * q_component)
        
        # Upsample and apply pulse shaping (rectangular pulse for now)
        # 上采样并应用脉冲成形（此处使用矩形脉冲）
        modulated_signal = np.repeat(symbols, self.samples_per_symbol)
        
        return modulated_signal
    
    def demodulate(self, received_signal: np.ndarray) -> np.ndarray:
        """
        Demodulate QPSK signal to bit stream / 将QPSK信号解调为比特流
        
        Uses coherent detection (assuming perfect carrier recovery).
        使用相干检测（假设理想载波恢复）。
        
        Parameters / 参数:
        ----------------
        received_signal : np.ndarray (complex)
            Received complex baseband signal / 接收的复数基带信号
            
        Returns / 返回:
        -------------
        bit_stream : np.ndarray (int, 0/1)
            Demodulated binary data / 解调后的二进制数据
        """
        # Downsample to symbol rate (take center sample of each symbol)
        # 下采样至符号率（取每符号中间采样点）
        symbols = received_signal[self.samples_per_symbol//2::self.samples_per_symbol]
        
        # Decision based on quadrants / 基于象限的判决
        # I > 0 → bit0=0, I < 0 → bit0=1
        # Q > 0 → bit1=0, Q < 0 → bit1=1
        bit0 = np.where(np.real(symbols) > 0, 0, 1)  # MSB
        bit1 = np.where(np.imag(symbols) > 0, 0, 1)  # LSB
        
        # Interleave bits: [b0_0, b1_0, b0_1, b1_1, ...]
        # 比特交织：[第0符号MSB, 第0符号LSB, 第1符号MSB, ...]
        bit_stream = np.empty(2 * len(symbols), dtype=int)
        bit_stream[0::2] = bit0
        bit_stream[1::2] = bit1
        
        return bit_stream
    
    def get_constellation(self) -> tuple:
        """
        Return ideal constellation points / 返回理想星座图点
        
        Returns / 返回:
        -------------
        constellation : tuple (i_points, q_points)
            I and Q coordinates of ideal constellation / 理想星座图的I、Q坐标
        """
        i_points = np.array([1, -1, -1, 1]) * self.amplitude
        q_points = np.array([1, 1, -1, -1]) * self.amplitude
        return (i_points, q_points)