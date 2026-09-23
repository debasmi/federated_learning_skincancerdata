## 📊 Dataset

**Source:** [MedMNIST+ on Zenodo](https://doi.org/10.5281/zenodo.10519652)  
**Version:** 3.0  
**Published:** January 16, 2024

[MedMNIST](https://medmnist.com/) is a collection of **18 standardized biomedical image datasets**, consisting of **12 two-dimensional (2D)** and **6 three-dimensional (3D)** datasets. The images are pre-processed to small standardized sizes and provided with classification labels, making them suitable for machine learning and deep learning experiments without requiring specialized medical background knowledge.

### 🩺 DermaMNIST

**DermaMNIST** is the dermatology/skin-lesion subset of MedMNIST used in this project. It consists of RGB dermatoscopic images categorized into **7 classes** for multi-class image classification.

| Item | Details |
|---|---|
| **File Used** | `dermamnist.npz` |
| **File Size** | ~19.7 MB |
| **Image Type** | RGB dermatoscopic images |
| **Image Dimensions** | 28 × 28 pixels |
| **Task** | Multi-class classification |
| **Number of Classes** | 7 |
| **Training Images** | 7,007 |
| **Validation Images** | 1,003 |
| **Test Images** | 2,005 |

### 📁 Dataset Structure

The dataset is provided as a NumPy `.npz` file containing the image data and corresponding labels for the training, validation, and test sets.

```text
dermamnist.npz
├── train_images
├── train_labels
├── val_images
├── val_labels
├── test_images
└── test_labels
```

To view the dataset kindly download the NPZ Viewer Extension in VS CODE or run the viewer.py file in order to see the visualized images.

## 📚 Citation

If you use the **DermaMNIST** dataset or other MedMNIST datasets in your research or project, please cite both of the following MedMNIST papers:

### 1. MedMNIST v2

> Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni.  
> **"MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification."**  
> *Scientific Data*, 2023.

```bibtex
@article{yang2023medmnist,
  title={MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
  author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
  journal={Scientific Data},
  year={2023}
}

@inproceedings{yang2021medmnist,
  title={MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis},
  author={Yang, Jiancheng and Shi, Rui and Ni, Bingbing},
  booktitle={IEEE 18th International Symposium on Biomedical Imaging (ISBI)},
  year={2021}
}
```
