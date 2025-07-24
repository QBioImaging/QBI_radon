[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GqOcOWj128oQ2ojBy8VX5bzg0zAY_MDz?usp=sharing)
# QBI_radon

**QBI_radon** is a Python library that provides an efficient, GPU-accelerated, and differentiable implementation of the **Radon transform** using **PyTorch ≥ 2.0**. 

The Radon transform maps an image to its Radon space representation — a key operation in solving **computed tomography (CT)** reconstruction problems. This library is designed to help researchers and developers combine **deep learning** and **model-based approaches** in a unified PyTorch framework.

---

## 🚀 Key Features

- ✅ **Differentiable Forward & Back Projections**  
  All transformations are fully compatible with PyTorch’s autograd system, allowing gradient computation via `.backward()`.

- ⚡ **Batch Processing & GPU Acceleration**  
  Designed for speed — supports batched operations and runs efficiently on GPUs. Faster than `skimage`'s Radon transform.

- 🔁 **Transparent PyTorch API**  
  Seamless integration with PyTorch pipelines. Compatible with **Nvidia AMP** for mixed-precision training and inference.

- 🧩 **Cross-Platform Support**  
  Built entirely on PyTorch ≥ 2.0, ensuring compatibility across major operating systems — Windows, Ubuntu, macOS, and more.

---

## 🧠 Applications

- Deep learning for CT image reconstruction  
- Model-based & hybrid inverse problems  
- Differentiable physics-based layers in neural networks  

---

## 🔧 Implemented Operations

- ✅ **Parallel Beam Projections**

Additional projection geometries and advanced features are under development. Stay tuned!

---

## 📦 Installation

```bash
pip install QBI-radon
```

## 🚀 Google Colab

You can try the library from your browser using Google Colab, you can find an example notebook [here](https://colab.research.google.com/drive/1GqOcOWj128oQ2ojBy8VX5bzg0zAY_MDz?usp=sharing).

## 📚 Citation
If you are using QBI_radon in your research, please cite the following paper:
<!-- 
```bibtex
@article{qbi_radon,
  title={QBI_radon: A PyTorch library for Radon transform},
  author={QBI_radon},
  journal={arXiv preprint arXiv:2507.12345},
  year={2025}
}
``` -->