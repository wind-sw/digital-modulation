"""
Hamming Error Correction Code (15,11) / 汉明纠错码(15,11)

Implements Hamming(15,11) linear block code capable of correcting single-bit errors.
实现可纠正单比特错误的汉明(15,11)线性分组码。
"""

import numpy as np


class Hamming1511:
    """
    Hamming(15,11) Error Correction Code / 汉明(15,11)纠错码
    
    A linear error-correcting code that encodes 11 data bits into 15 bits 
    by adding 4 parity bits. Can detect and correct any single-bit error.
    
    线性纠错码，通过添加4个校验位将11个数据位编码为15位。
    可检测并纠正任何单比特错误。
    
    Code parameters / 码参数:
    ----------------------
    - Block length n = 15 / 码长n=15
    - Message bits k = 11 / 信息位k=11  
    - Parity bits n-k = 4 / 校验位4
    - Error correction capability: t = 1 / 纠错能力t=1
    - Code rate: R = 11/15 ≈ 0.733 / 码率R=11/15
    """
    
    def __init__(self):
        """
        Initialize Hamming(15,11) codec / 初始化编解码器
        
        Generator matrix G and parity-check matrix H are defined according 
        to standard Hamming(15,11) construction.
        """
        # Parity check matrix H (4x15) / 校验矩阵H
        # Columns are binary representations of 1-15 / 列为1-15的二进制表示
        self.parity_check_matrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1]
        ], dtype=int)
        
        # Generator matrix G (11x15) / 生成矩阵G
        # Systematic form: [I_11 | P] / 系统形式：[I_11 | P]
        self.generator_matrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1]
        ], dtype=int)
        
        # Syndrome table for error correction / 伴随式查表纠错
        # Maps syndrome (as integer) to error position / 将伴随式（整数）映射到错误位置
        self.syndrome_table = self._build_syndrome_table()
    
    def _build_syndrome_table(self):
        """
        Build syndrome-to-error-position lookup table / 构建伴随式-错误位置查找表
        """
        table = {}
        for i in range(15):
            error_pattern = np.zeros(15, dtype=int)
            error_pattern[i] = 1
            syndrome = np.dot(self.parity_check_matrix, error_pattern) % 2
            syndrome_int = int(np.packbits(syndrome, bitorder='big')[0])
            table[syndrome_int] = i
        # Syndrome 0 = no error / 伴随式0=无错误
        table[0] = -1
        return table
    
    def encode(self, data_bits):
        """
        Encode 11-bit data to 15-bit codeword / 将11位数据编码为15位码字
        
        Parameters / 参数:
        ----------------
        data_bits : np.ndarray (int, 0/1)
            Input data bits (length multiple of 11) / 输入数据比特（长度为11的倍数）
            
        Returns / 返回:
        -------------
        codeword : np.ndarray (int, 0/1)
            Encoded codeword (length multiple of 15) / 编码后的码字（长度为15的倍数）
        """
        data_bits = np.asarray(data_bits, dtype=int)
        
        # Pad to multiple of 11 / 补零至11的倍数
        remainder = len(data_bits) % 11
        if remainder != 0:
            padding = 11 - remainder
            data_bits = np.concatenate([data_bits, np.zeros(padding, dtype=int)])
        
        num_blocks = len(data_bits) // 11
        codeword = np.zeros(num_blocks * 15, dtype=int)
        
        for i in range(num_blocks):
            block = data_bits[i*11:(i+1)*11]
            # Matrix multiplication: c = d * G (mod 2)
            # 矩阵乘法：c = d * G (模2)
            encoded = np.dot(block, self.generator_matrix) % 2
            codeword[i*15:(i+1)*15] = encoded
            
        return codeword
    
    def decode(self, received_bits):
        """
        Decode and correct errors / 解码并纠正错误
        
        Uses syndrome decoding to locate and correct single-bit errors.
        使用伴随式译码定位并纠正单比特错误。
        
        Parameters / 参数:
        ----------------
        received_bits : np.ndarray (int, 0/1)
            Received codeword (length multiple of 15) / 接收码字（长度为15的倍数）
            
        Returns / 返回:
        -------------
        decoded_data : np.ndarray (int, 0/1)
            Decoded data bits (length multiple of 11) / 解码后的数据比特（长度为11的倍数）
        """
        received_bits = np.asarray(received_bits, dtype=int)
        
        # Pad to multiple of 15 / 补零至15的倍数
        remainder = len(received_bits) % 15
        if remainder != 0:
            padding = 15 - remainder
            received_bits = np.concatenate([received_bits, np.zeros(padding, dtype=int)])
        
        num_blocks = len(received_bits) // 15
        decoded_data = np.zeros(num_blocks * 11, dtype=int)
        
        for i in range(num_blocks):
            block = received_bits[i*15:(i+1)*15]
            
            # Calculate syndrome: s = H * r^T (mod 2)
            # 计算伴随式：s = H * r^T (模2)
            syndrome = np.dot(self.parity_check_matrix, block) % 2
            syndrome_int = int(np.packbits(syndrome, bitorder='big')[0])
            
            # Correct error if syndrome non-zero / 若伴随式非零则纠错
            if syndrome_int in self.syndrome_table and self.syndrome_table[syndrome_int] != -1:
                error_pos = self.syndrome_table[syndrome_int]
                block[error_pos] = 1 - block[error_pos]  # Flip bit / 翻转比特
            
            # Extract data bits (first 11 bits in systematic code)
            # 提取信息位（系统码的前11位）
            decoded_data[i*11:(i+1)*11] = block[:11]
            
        return decoded_data