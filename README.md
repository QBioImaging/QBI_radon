[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GqOcOWj128oQ2ojBy8VX5bzg0zAY_MDz?usp=sharing)
# 📦 QBI_radon

**QBI_radon** is a Python library that provides an efficient, GPU-accelerated, and differentiable implementation of the **Radon transform** using **PyTorch ≥ 2.0**.

QBI_radon provides **GPU-accelerated forward and backward projection operations** for tomography, making it ideal for computed tomography (CT) research and development.

The Radon transform maps an image to its Radon space representation — a key operation in solving **CT reconstruction problems**. This GPU-accelerated library is designed to help researchers and developers obtain **fast and accurate tomographic reconstructions**, and seamlessly combine **deep learning** and **model-based approaches** in a unified PyTorch framework.

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
- GPU-accelerated Filtered Backprojection


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
If you are using QBI_radon in your research, please cite the following:

[![DOI](https://zenodo.org/badge/811419352.svg)](https://doi.org/10.5281/zenodo.16416058)

```bibtex
@software{Trinh_QBioImaging_QBI_radon_2025,
author = {Trinh, Minh-Nhat and Teresa, M Correia},
doi = {https://doi.org/10.5281/zenodo.16416059},
month = jul,
title = {{QBioImaging/QBI\_radon}},
url = {https://github.com/QBioImaging/QBI_radon},
version = {v1.7},
year = {2025}
}
```

<!-- ## 📝 Acknowledgements
This study received Portuguese national funds from FCT—Foundation for Science and Technology through projects UIDB/04326/2020 (DOI:https://doi.org/10.54499/UIDB/04326/2020), UIDP/04326/2020 (DOI:https://doi.org/10.54499/UIDP/04326/2020) and LA/P/0101/2020 (DOI:https://doi.org/10.54499/LA/P/0101/2020). This Project received funding from ‘la Caixa’ Foundation and FCT, I P under the Project code LCF/PR/HR22/00533, European Union’s Horizon 2020 research and innovation program under the Marie Skłodowska-Curie OPTIMAR grant with agreement no 867450 (DOI:https://doi.org/10.3030/867450), European Union’s Horizon Europe Programme IMAGINE under grant agreement no. 101094250
(DOI:https://doi.org/10.3030/101094250), and NVIDIA GPU hardware grant. -->
