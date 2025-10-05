# Image Translation: GoogleMap ←→ SatelliteImages
## 🔍 Overview
This project explores image-to-image translation between Google Maps 🗺️ and Satellite Images 🛰️ using Conditional Generative Adversarial Networks (cGANs).
By leveraging a Pix2Pix architecture, the model learns to generate high-fidelity satellite images from map views—and vice versa—capturing spatial details and realistic textures.


## 🛰️ About

In real-world applications, **satellite image analysis** plays a crucial role in areas such as **environmental monitoring**, **urban planning**, and **disaster management**.  

This project aims to **bridge the gap between map and satellite domains** using a **Conditional Generative Adversarial Network (cGAN)** framework.

### 💡 Model Overview

- **Generator:** U-Net architecture for effective feature encoding and decoding  
- **Discriminator:** PatchGAN architecture for local realism judgment  
- **Framework:** TensorFlow / Keras  


### 🔄 Input & Output

| Input | Output | Description |
|--------|---------|-------------|
| Google Map 🗺️ | Satellite Image 🛰️ | Converts map view into realistic satellite imagery |
| Satellite Image 🛰️ | Google Map 🗺️ | Translates satellite view into styled Google Maps |



## 🛠️ Requirements

### ⚙️ Environment Information

| Component | Version / Info |
|------------|----------------|
|  **Python** | 3.8.10 |
|  **TensorFlow** | 2.8.0 |
|  **Keras** | 2.8.0 |
|  **GPU** | NVIDIA RTX 4090 |
|  **CUDA** | Available |

😃 Make sure you have the above Python libraries and environments before running the notebooks!

## 📘 Start with This Project!

### 📂 Dataset 
### The dataset is located in the `maps/` directory and contains:

- **Training set:** 1,096 images

- **Validation set:** 1,098 images

### Each image is composed of two halves:

- **Left:** Satellite Image 🛰️

- **Right:** Corresponding Google Map 🗺️

These paired images represent the same geographic location. Show as in [image.png]



### Preprocess:
Before training, split each composite image into two separate images:

```
sat_img = pixels[:, :256]     # Left half → Satellite image
map_img = pixels[:, 256:]     # Right half → Google Map image
```
## 🧠 Model Architecture

### 🧩 Generator — U-Net (Encoder–Decoder)

- **Downsampling (Encoder):** series of Conv2D + BatchNorm + LeakyReLU layers  
- **Bottleneck:** compresses spatial features for translation learning  
- **Upsampling (Decoder):** Conv2DTranspose + skip connections to preserve spatial details  
- **Activation:** Tanh (outputs normalized to [-1, 1])  

### 🧱 Discriminator — PatchGAN

- Evaluates local patches of the input pair (real or generated)  
- Learns to classify whether patches of the generated image are realistic  
- Uses Conv2D + LeakyReLU + BatchNorm layers with a final Sigmoid activation  

### ⚙️ Conditional GAN Integration

- The **generator** produces translated images (e.g., map → satellite).  
- The **discriminator** assesses whether these images are real or fake.  
- **Combined training uses:**  
  - **Adversarial loss:** Binary cross-entropy  
  - **L1 loss:** Mean absolute error between generated and real images  
  - **Loss weights:** `[1, 100]` (as suggested in the Pix2Pix paper)  

---

## 🚀 Training

Two Jupyter notebooks are provided for bidirectional translation tasks:

| Notebook | Task | Description |
|----------|------|-------------|
| `pix2pix_Map2Satellite.ipynb` | 🗺️ → 🛰️ | Convert Google Maps into corresponding satellite images |
| `pix2pix_Satellite2Map.ipynb` | 🛰️ → 🗺️ | Convert satellite images into Google Maps |

Training was performed on an **NVIDIA RTX 4090 GPU**, requiring approximately **30–40 minutes per task**.

---

## 🧩 Pretrained Models

For quick verification and testing, two pretrained models are provided:

| Model File | Task | Direction |
|------------|------|-----------|
| `Map2SatModel.h5` | Map → Satellite | 🗺️ ➡️ 🛰️ |
| `Satellite2Map.h5` | Satellite → Map | 🛰️ ➡️ 🗺️ |

You can load and test them directly without retraining.
