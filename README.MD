# Digital Communication System Simulation / 数字通信系统仿真

<p align="center">
  <img src="docs/diagram.png" alt="System Block Diagram" width="600"/>
</p>

A comprehensive Python-based simulation framework for digital communication systems, covering the complete transmission chain from source quantization to channel decoding.

基于 Python 的数字通信系统综合仿真框架，涵盖从信源量化到信道解码的完整传输链路。

---

## Acknowledgments / 致谢

This project is forked and enhanced from the original [alicescfernandes/digital-modulation: Simulation of a communication channel written in Python](https://github.com/alicescfernandes/digital-modulation) repository. Special thanks to the original author for establishing the foundational architecture and educational approach to digital communications simulation.

本项目 fork 自原 [alicescfernandes/digital-modulation: Simulation of a communication channel written in Python](https://github.com/alicescfernandes/digital-modulation) 仓库。特别感谢原作者奠定了数字通信仿真的基础架构与教学理念。

---

## System Architecture / 系统架构

The simulation implements a complete end-to-end digital communication pipeline:

本仿真实现了完整的端到端数字通信流程：

```
Analog Signal → Quantization → Source Coding → Channel Coding → Modulation → Channel → Demodulation → Decoding → Reconstruction
模拟信号 → 量化 → 信源编码 → 信道编码 → 调制 → 信道 → 解调 → 解码 → 重建
```

### Core Modules / 核心模块

| Module | Description | 描述 |
|--------|-------------|------|
| `quantizer.py` | Uniform Quantizer (Mid-rise/Mid-tread) | 均匀量化器（中升型/中平型） |
| `converter.py` | ADC/DAC (Level-to-Bits conversion) | 模数/数模转换（电平-比特转换） |
| `error_correction.py` | Hamming(15,11) FEC | 汉明(15,11)前向纠错编码 |
| `modulator.py` | QPSK Modulation/Demodulation | QPSK 调制解调 |
| `channel.py` | AWGN Channel with Header Protection | 带头部保护的 AWGN 信道 |
| `utils.py` | SNR/BER calculation utilities | 信噪比/误码率计算工具 |
| `simulate.py` | End-to-end simulation & visualization | 端到端仿真与可视化 |

---

## Features / 功能特性

### 1. Complete Transmission Chain / 完整传输链路
- **Quantization**: Configurable bit-depth (R) with theoretical SNR calculation
- **量化**: 可配置比特深度(R)，含理论信噪比计算
- **Source Coding**: Integer-to-binary packing/unpacking
- **信源编码**: 整数到二进制打包/解包
- **Channel Coding**: Hamming(15,11) single-error correction
- **信道编码**: 汉明(15,11)单比特纠错
- **Modulation**: QPSK with Gray coding and constellation visualization
- **调制**: 格雷编码 QPSK 及星座图可视化
- **Channel**: AWGN with adjustable noise power and header protection
- **信道**: 可调噪声功率及头部保护的加性高斯白噪声

### 2. Image Transmission Simulation / 图像传输仿真
Simulate multimedia data transmission with visual output comparison:

多媒体数据传输仿真，提供可视化输出对比：

- Side-by-side comparison: Original | With FEC | Without FEC
- 并排对比：原始图像 | 带纠错 | 无纠错
- Constellation diagrams for channel quality analysis
- 星座图用于信道质量分析
- Automatic metrics: BER (pre/post correction), SNR at each stage
- 自动指标：纠错前后误码率、各阶段信噪比

### 3. Performance Metrics / 性能指标
- **BER Calculation**: Theoretical (QPSK) vs Practical (Bit-by-bit comparison)
- **误码率计算**: 理论值(QPSK) vs 实际值(逐比特对比)
- **SNR Monitoring**: Quantization SNR, Channel SNR, Reception SNR
- **信噪比监测**: 量化信噪比、信道信噪比、接收信噪比
- **Error Correction Gain**: Quantified improvement with Hamming codes
- **纠错增益**: 汉明码带来的性能提升量化

---

## Installation / 安装

```bash
# Clone the repository / 克隆仓库
git clone <repository-url>
cd digital-modulation

# Install dependencies / 安装依赖
pip install numpy matplotlib pillow scipy

```

**Requirements / 环境要求:**
- Python 3.7+
- NumPy
- Matplotlib (for visualization / 可视化)
- Pillow (for image processing / 图像处理)
- SciPy (for BER calculation / 误码率计算)

---

## Usage / 使用方法

### Basic Simulation / 基础仿真

Run the main simulation with default parameters (Lena image test pattern):

使用默认参数运行主仿真（Lena图像测试模式）：

```bash
python simulate.py
```

### Parameter Configuration / 参数配置

Modify the test cases in `simulate.py`:

修改 `simulate.py` 中的测试用例：

```python
# Format: (noise_power, bit_depth)
# 格式：(噪声功率, 量化比特数)
test_cases = [
    (0.05, 8),   # Low noise, 8-bit quantization / 低噪声，8位量化
    (0.1, 8),    # Medium noise / 中等噪声
    (0.5, 6),    # High noise, 6-bit quantization / 高噪声，6位量化
]
```

### Custom Image Input / 自定义图像输入

```python
from simulate import process_image_transmission

# Process custom image with specific parameters
# 使用特定参数处理自定义图像
process_image_transmission(
    noise_level=0.5,      # AWGN noise power / AWGN噪声功率
    bit_depth=8,          # Quantization bits / 量化比特数
    image_path="path/to/your/image.jpg"
)
```

---

## Output Examples / 输出示例

### Image Transmission Results / 图像传输结果

```
generated/
├── image_with_fec_R8_N0.05.png      # Reconstructed with FEC / 带纠错重建
├── image_no_fec_R8_N0.05.png        # Reconstructed without FEC / 无纠错重建
├── comparison_R8_N0.05.png          # Side-by-side comparison / 并排对比图
└── constellation_R8_N0.05.png       # QPSK constellation / QPSK星座图
```

### Console Metrics / 控制台指标

```
Metrics Summary / 性能指标摘要:
  Parameter                         With FEC       Without FEC
  --------------------------------- --------------- ---------------
  Quantization SNR (theory)                 dB
  Channel SNR                               dB          dB
  Reception SNR                             dB          dB
  BER before correction                            
  BER after correction                             
```

---

## Technical Details / 技术细节

### Quantization / 量化
- **Mid-rise**: Uniform steps without zero level, optimal for AC signals
- **中升型**: 无零电平的均匀步长，适用于交流信号
- **Mid-tread**: Includes zero level, suitable for signals with DC component
- **中平型**: 包含零电平，适用于含直流分量的信号

### QPSK Modulation / QPSK调制
- Gray coding to minimize bit errors per symbol error
- 格雷编码以最小化每个符号错误的比特错误数
- Square-root raised cosine pulse shaping (rectangular in current implementation)
- 平方根升余弦脉冲成形（当前实现为矩形脉冲）
- Coherent detection with perfect carrier recovery assumption
- 相干检测，假设理想载波恢复

### Hamming(15,11) FEC / 汉明纠错
- Code rate: 11/15 ≈ 0.733
- 码率：11/15 ≈ 0.733
- Single-bit error correction capability
- 单比特纠错能力
- Systematic encoding: Data bits + Parity bits
- 系统码编码：数据位 + 校验位

### AWGN Channel / AWGN信道
- Complex baseband noise model
- 复数基带噪声模型
- Configurable noise power (variance σ²)
- 可配置噪声功率（方差σ²）
- Header protection mode for synchronization/pilot simulation
- 头部保护模式用于同步/导频仿真

---

## Educational Value / 教育价值

This simulation serves as a practical tool for understanding:

本仿真可作为理解以下内容的实践工具：

- **Quantization Effects**: Visualize bit-depth impact on image quality
- **量化效应**: 可视化比特深度对图像质量的影响
- **Error Correction**: Compare images with/without FEC under same noise conditions
- **纠错编码**: 在相同噪声条件下对比有无FEC的图像
- **Modulation Impairments**: Observe constellation spreading with increasing noise
- **调制损伤**: 观察噪声增加时的星座图扩散
- **End-to-End Tradeoffs**: Balance between quantization precision, bandwidth (code rate), and power (SNR)
- **端到端权衡**: 量化精度、带宽（码率）和功率（信噪比）之间的平衡

---

## Contributing / 贡献

Contributions are welcome! Areas for improvement:

欢迎贡献！以下是可改进方向：

- [ ] Vectorization of loop operations for performance
- [ ] 循环操作向量化以提升性能
- [ ] Additional modulation schemes (16-QAM, FSK)
- [ ] 更多调制方式（16-QAM、FSK）
- [ ] Audio signal processing support
- [ ] 音频信号处理支持
- [ ] Differential PCM (DPCM) implementation
- [ ] 差分脉冲编码调制(DPCM)实现
- [ ] Additional CRC and error correction codes
- [ ] 更多CRC及纠错码

---

*For educational purposes in digital communications and signal processing.*
*用于数字通信与信号处理学习目的。*
*This is written by KIMI*