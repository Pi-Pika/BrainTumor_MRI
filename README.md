# 🧠 Brain Tumor Segmentation & Classification (Joint U-Net Framework)

This project implements a **joint deep learning framework** for:

- 🩺 Brain tumor segmentation (binary mask prediction)
- 🧬 Brain tumor classification (Glioma, Meningioma, Pituitary, No
  Tumor)

The model is built using a custom U-Net architecture with optional
Attention Gates and trained in a multi-task learning setup.

---

## 🚀 Key Features

- Custom U-Net implementation from scratch
- Optional Attention U-Net
- Joint segmentation + classification training
- Albumentations-based preprocessing
- Comprehensive evaluation metrics (IoU, Dice, Accuracy, Precision,
  Recall, F1)
- Google Colab compatible

---

## 📂 Dataset Structure

dataset_root/ └── segmentation_task/ ├── train/ │ ├── images/ │ └──
masks/ ├── val/ │ ├── images/ │ └── masks/ └── test/ ├── images/ └──
masks/

Image filenames must contain class identifiers: - *gl* → Glioma - *me* →
Meningioma - *pi* → Pituitary - *no* → No Tumor

---

## 🏗 Model Architecture

The model outputs:

- Segmentation mask logits
- Classification logits (4 classes)

Loss Function: Total Loss = BCEWithLogitsLoss (Segmentation) +
CrossEntropyLoss (Classification)

---

## 📊 Evaluation Metrics

Segmentation: - IoU (Jaccard Index) - Dice Score - Binary Accuracy

Classification: - Accuracy - Precision (macro) - Recall (macro) - F1
Score (macro)

---

## ⚙️ Hyperparameters

- IMG_SIZE = 256
- BATCH_SIZE = 16
- LEARNING_RATE = 1e-4
- EPOCHS = 20
- DEVICE = cuda if available else cpu

---

## 🛠 Installation

pip install torch torchvision albumentations torchmetrics opencv-python
numpy pandas matplotlib seaborn scikit-learn tqdm

---

## 🧪 Training Example

model = CustomUNet(attention=False).to(DEVICE)

history = train_model( model, seg_train_loader, cls_train_loader,
val_loader, epochs=10 )

---

## 💾 Save & Load Model

Save: torch.save(model.state_dict(), "model_name.pth")

Load: model = CustomUNet(attention=True).to(DEVICE)
model.load_state_dict(torch.load("model_name.pth", map_location='cpu'))
model.eval()

---

## 🔁 Reproducibility

seed_everything(seed=42)

Ensures deterministic behavior across runs.

---

## 📈 Future Improvements

- Add pretrained encoder backbone
- Hybrid Dice + BCE loss
- Grad-CAM visualization
- Streamlit deployment
- ONNX export for inference

---

## 👤 Author

Mahmudul Hasan Piash
Student \| Engineering & AI Research
