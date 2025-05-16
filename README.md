
# Enhanced Super-Resolution using ESRGAN 🚀🖼️

This project implements **ESRGAN (Enhanced Super-Resolution Generative Adversarial Network)** for improving image resolution. By leveraging adversarial training, perceptual loss, and Residual-in-Residual Dense Blocks (RRDB), this model reconstructs high-resolution images from low-resolution inputs with impressive perceptual quality.

![Image](https://github.com/user-attachments/assets/8f15288a-7ef1-48cd-83a1-16a2e5edd346)---

## 📌 Objective

To enhance the resolution of low-quality images using a deep learning-based ESRGAN model and compare its performance against traditional CNN-based super-resolution techniques.

---

## 🧠 Approach

- **Dataset Used**: [DIV2K Dataset](https://www.kaggle.com/datasets/joe1995/div2k-dataset)
- **Architecture**:
  - **Generator**: Built using Residual-in-Residual Dense Blocks (RRDB), followed by upsampling layers.
  - **Discriminator**: CNN-based model trained to distinguish real vs. generated high-resolution images.
  - **Loss Functions**:
    - L1 Loss for pixel-level accuracy
    - VGG Perceptual Loss for visual quality
    - Adversarial Loss for GAN stability
- **Training Strategy**:
  - Trained for 100 epochs using Adam optimizers
  - Evaluation using PSNR, SSIM, Precision, Recall, and F1 Score

---

## 🛠️ Tools & Libraries

- Python
- PyTorch
- Albumentations
- OpenCV
- Matplotlib
- scikit-image / sklearn

---

## 📊 Results

| Metric       | CNN Model | ESRGAN Model |
|--------------|-----------|--------------|
| PSNR         | 7.09 dB   | 25.71 dB     |
| SSIM         | 0.18      | 0.93         |

✔️ Visual comparison also confirms that ESRGAN generates more realistic and sharper textures.

1.
![Image](https://github.com/user-attachments/assets/c846d651-b222-4266-bb5a-fe9be9eb431a)
2.
![Image](https://github.com/user-attachments/assets/28df80a3-e0f0-4455-bfae-cccdc6b5461a)

---

## 📈 Key Features

- Achieved **~250% improvement in PSNR** over basic CNN-based models.
- Implemented custom dataloaders, loss functions, and evaluation metrics.
- Created visual tools for evaluating and comparing outputs (e.g., histograms, PR curves, ROC).

---

## 📂 Folder Structure

```
├── ESRGAN.ipynb            # project file
├── README.md               # Project Overview
```

---

## 📌 Future Work

- Add support for real-time super-resolution on webcam inputs
- Convert the model to ONNX/TensorRT for deployment
- Extend ESRGAN with attention mechanisms or Swin Transformers



## 📬 Contact

**Karan**  
Computer Science Student | Passionate about AI/ML  
📧 Reach me via [LinkedIn](https://www.linkedin.com) or [Email](mailto:your@email.com)
