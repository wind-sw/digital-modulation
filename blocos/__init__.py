"""
Digital Communication System Blocks / 数字通信系统模块集

This package contains modular implementations of digital communication 
components including quantization, modulation, channel simulation, 
and error correction coding.

本包包含数字通信组件的模块化实现，包括量化、调制、信道仿真和纠错编码。
"""

from .converter import ADConverter
from .quantizer import UniformQuantizer
from .modulator import QPSKModulator
from .channel import AWGNChannel
from .error_correction import Hamming1511
from .utils import *

__all__ = [
    'ADConverter',
    'UniformQuantizer', 
    'QPSKModulator',
    'AWGNChannel',
    'Hamming1511',
    'calculate_signal_power',
    'calculate_uniform_quantization_snr',
    'calculate_practical_snr',
    'calculate_qpsk_ber',
    'calculate_bit_error_rate',
    'calculate_noise_psd'
]