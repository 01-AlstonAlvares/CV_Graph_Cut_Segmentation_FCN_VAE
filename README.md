# 🧠 Computer Vision Assignment: GrabCut, FCN, and VAE (Analysis) by st126488

This repository contains the **project assignment**, **setup instructions**, and **full analysis** for all three tasks — presented without Python code blocks.

---

## 📘 README: Project Overview & Setup

### 📋 Table of Contents

1. [🛠️ Setup and Installation](#-setup-and-installation)  
2. [🚀 How to Run](#-how-to-run)  
3. [📒 Task Analysis](#-task-analysis)  
   - [Task 1: Graph Cut Segmentation](#task-1-graph-cut-segmentation)  
   - [Task 2: Fully Convolutional Network (FCN)](#task-2-fully-convolutional-network-fcn)  
   - [Task 3: Variational Autoencoder (VAE)](#task-3-variational-autoencoder-vae)

---

## 🛠️ Setup and Installation

This project is built in **Python** and uses standard libraries from the **scientific computing** and **deep learning** ecosystem.

### 1. Create a Virtual Environment (Recommended)

It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts with other projects.

```bash
# Create a new virtual environment
python -m venv venv

# Activate the environment
# On Windows (cmd.exe)
.\venv\Scripts\activate

# On macOS/Linux (bash)
source venv/bin/activate
```

### 2. Install Required Libraries

Install all necessary libraries using `pip`:

```bash
pip install jupyterlab opencv-python torch torchvision numpy matplotlib pandas requests
```

#### Libraries Breakdown

| Library | Purpose |
|----------|----------|
| **jupyterlab** | For running notebooks interactively. |
| **opencv-python** | Used in Task 1 for `cv2.grabCut` and DNN object detection. |
| **torch**, **torchvision** | Deep learning framework for Task 2 (FCN) and Task 3 (VAE). |
| **numpy**, **matplotlib** | Numerical operations and plotting. |
| **pandas** | Comparison table display in Task 2. |
| **requests** | Downloads model files and sample images automatically. |

---

## 🚀 How to Run

After installing the libraries, you can run any of the tasks.

1. Make sure your virtual environment is activated.  
2. Start JupyterLab or Notebook:

```bash
jupyter lab
# or
jupyter notebook
```

3. Create new notebooks:
   - `Task_1_GraphCut.ipynb`
   - `Task_2_FCN.ipynb`
   - `Task_3_VAE.ipynb`

4. Copy the respective task’s code and run all cells from top to bottom.

---

## 📒 Task Analysis

---

### 🧩 Task 1: Graph Cut Segmentation

**Objective:**  
Implement graph-based image segmentation using `cv2.grabCut` guided by an automatic bounding box.

#### Steps
1. **Setup:** Import libraries and download model/sample images.  
2. **Object Detection:** Detect person in image using MobileNet-SSD.  
3. **GrabCut Implementation:** Run `cv2.grabCut` for different iterations.  
4. **Execution & Visualization:** Run pipeline for 1, 3, and 5 iterations.  
5. **Analysis:** Perform qualitative and quantitative analysis.

---

#### 🧠 Qualitative Analysis

Using sample images (e.g., surfer `asm-1.jpg` and sleeping man `asm-2.jpg`):

- **1 Iteration:** Rough segmentation; may include large background portions.  
- **3 Iterations:** Improved segmentation; clearer edges and background removal.  
- **5 Iterations:** Most refined; subtle improvements, showing convergence.

#### 📊 Quantitative Analysis

Foreground pixel count decreases as iterations increase — showing that the algorithm progressively removes background pixels and converges after around 3–5 iterations.

**Conclusion:**  
GrabCut progressively refines its segmentation mask, with diminishing returns beyond 3 iterations.

---

### 🧠 Task 2: Fully Convolutional Network (FCN)

**Objective:**  
Implement an **FCN-32s** model for semantic segmentation using a synthetic dataset, comparing **Transpose Convolution** vs **Bilinear Upsampling**.

#### Steps
1. Setup device and imports.  
2. Create synthetic dataset (geometric shapes).  
3. Implement FCN-32s model with pre-trained **VGG16** backbone.  
4. Add **Pixel Accuracy** and **Mean IoU** metrics.  
5. Train and evaluate using:
   - **Experiment 1:** Transpose Convolution  
   - **Experiment 2:** Bilinear Upsampling  
6. Analyze and visualize results.

---

#### ⚙️ Comparison: Transpose Convolution vs. Bilinear Upsampling

| Method | Description | Pros | Cons |
|--------|--------------|------|------|
| **nn.ConvTranspose2d** | Learned upsampling | Learns optimal upsampling; potentially sharper results | Adds parameters; can cause checkerboard artifacts |
| **nn.Upsample (bilinear)** | Fixed interpolation | Simple, fast, zero parameters | May produce slightly blurrier results |

#### 📈 Analysis of Results

- Both converge quickly with high accuracy and mIoU.  
- **Bilinear Upsampling** achieves smooth, stable convergence and performs equally well on simple datasets.  
- **Transpose Convolution** provides flexibility for complex datasets.

**Conclusion:**  
For simple tasks → Bilinear is efficient and sufficient.  
For complex real-world datasets → Transpose Convolution may outperform due to learnable upsampling.

---

### 🔮 Task 3: Variational Autoencoder (VAE)

**Objective:**  
Implement a **VAE** on MNIST to learn a latent representation, generate digits, and analyze the impact of latent dimensionality.

#### Steps
1. Load and preprocess MNIST.  
2. Define Encoder, Decoder, and Reparameterization functions.  
3. Define loss: **Reconstruction (BCE)** + **KL Divergence (KLD)**.  
4. Train and visualize latent space.  
5. Run two experiments:
   - `latent_dim = 128`
   - `latent_dim = 256`

---

#### 📉 VAE Loss Analysis

A typical loss (~98.7) is reasonable because:
- **Reconstruction Loss (BCE):** Summed over all 784 pixels.  
- **KL Divergence (KLD):** Summed over latent dimensions.  

Loss decreasing steadily (e.g., from 165.8 → 98.7) indicates proper learning.

---

#### 🧩 Comparison: Latent Dim = 128 vs. 256

| Aspect | Latent Dim 128 | Latent Dim 256 |
|---------|----------------|----------------|
| **Reconstruction Quality** | Slightly blurrier but well-structured | Sharper reconstructions |
| **Generated Images** | More "average" digits | Greater variety of styles |
| **Latent Space** | More compressed, smoother | Less regularized but richer |
| **Overall** | Efficient, well-structured latent space | Higher fidelity but less compression |

---

#### 🏁 Overall Conclusion

- Increasing **latent dimension** improves reconstruction but reduces latent regularity.  
- For **MNIST**, both 128 and 256 dimensions work well.  
- Trade-off: **Compression vs. Fidelity** — smaller latent spaces yield smoother, more interpretable representations.

---

## 🧾 Summary

| Task | Technique | Key Insight |
|------|------------|-------------|
| **1. GrabCut** | Graph-based segmentation | Iterative refinement improves edge accuracy. |
| **2. FCN** | Semantic segmentation | Bilinear upsampling is efficient for simple data. |
| **3. VAE** | Generative modeling | Larger latent spaces improve reconstruction but reduce structure. |

---
