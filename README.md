# MVP: Mamba Visual Perceptron for Autonomous Flight Semantic Segmentation

**Real-time Semantic Segmentation for Intelligent Autonomous Aircraft Systems**

This repository contains the official PyTorch implementation of the paper:

> *"Mamba Visual Perceptron: A Real-time Semantic Segmentation Framework for Autonomous Flight Perception"*

---

## 🚀 Overview

With the increasing severity of urban congestion, autonomous flight technology has become a promising solution for alleviating ground traffic pressure. However, the application of environmental perception algorithms in the autonomous flight domain remains relatively scarce, as aircraft face unique challenges including complex airflows, wake vortices, and high-speed variations that require reliable visual perception systems capable of real-time processing.

To address this issue, this paper proposes an intelligent technology system consisting of four modules: **Perception**, **Decision-making**, **Action**, and **Feedback**. Within this framework, we introduce the **Mamba Visual Perceptron (MVP)** model for real-time semantic segmentation.

---

## 📂 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Autonomous Flight System                   │
├─────────────────────────────────────────────────────────────┤
│  Perception          │  MVP Model (Real-time Segmentation) │
│  Decision-making     │  Intelligent Navigation              │
│  Action              │  Flight Control                       │
│  Feedback            │  Performance Optimization             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧩 Model Architecture

The **MVP (Mamba Visual Perceptron)** model combines:

- **ViT-Backbone**: Vision Transformer for feature extraction
- **CNN Branch**: Local feature enhancement
- **Mamba Block**: Long-range dependency modeling with linear complexity
- **Segmentation Head**: Multi-scale feature upsampling

---

## 📊 Performance

Evaluated on the **CFD Simulation Dataset** for aircraft wake vortex detection:

| Metric | Value |
|--------|-------|
| **mIoU** | 95.62% |
| **PA (Pixel Accuracy)** | 97.64% |

### Comparison with State-of-the-Art Methods

| Model | mIoU | PA |
|-------|------|-----|
| **MVP (Ours)** | **95.62%** | **97.64%** |
| U-Net | 87.23% | 89.45% |
| DeepLabv3+ | 89.56% | 91.78% |
| SegNet | 85.67% | 88.12% |
| LinkNet | 84.23% | 86.89% |
| PSPNet | 88.91% | 90.56% |

MVP outperforms **19 state-of-the-art segmentation models**.

---

## 📂 Dataset

### CFD Simulation Dataset

To address the data scarcity issue in aviation, we construct a **Computational Fluid Dynamics (CFD) Simulation Dataset** for aircraft wake vortex detection.

The dataset includes:
- Synthetic wake vortex images generated via CFD simulation
- High-fidelity airflow visualization
- Multiple aircraft configurations

> Due to copyright and licensing restrictions, we do not redistribute raw simulation data. Please contact the authors for dataset access.

---

## 🔬 Key Features

- **Real-time Processing**: Efficient inference for autonomous flight applications
- **Mamba Architecture**: O(n) linear complexity for long-range dependencies
- **3D Attention Visualization**: Enhanced model interpretability via 3D attention weight projection
- **Robust Generalization**: Validated on both CFD simulation and real-world images

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/MVP.git
cd MVP
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.1+ (for GPU support)
- torchvision
- mamba-ssm
- matplotlib

---

## 📖 Usage

### Training

```bash
python train.py --dataset ./data/cfd_dataset --model mvp --epochs 100
```

### Evaluation

```bash
python evaluate.py --checkpoint ./checkpoints/mvp_cfd.pth --dataset ./data/test
```

### Inference

```python
from model import SegModule

model = SegModule(num_classes=4).cuda()
model.load_state_dict(torch.load('mvp_cfd.pth'))
model.eval()

# Input: (B, 3, H, W)
output = model(input_image)
```

---

## 🔔 Visualization

We provide comprehensive visualization tools:

- **Grad-CAM Comparison**: Feature attribution analysis
- **3D Attention Weight Projection**: Multi-dimensional attention visualization
- **t-SNE Feature Distribution**: Learned feature embedding analysis

---

## 📝 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{mvp2026,
  title={Mamba Visual Perceptron: A Real-time Semantic Segmentation Framework for Autonomous Flight Perception},
  author={Your Name},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📧 Contact

For questions, issues, or collaboration inquiries, please open an issue or contact:

- Email: your.email@example.com
- GitHub Issues: [Open an Issue](https://github.com/yourusername/MVP/issues)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgment

We thank the open-source community for providing excellent tools and datasets that made this work possible.
