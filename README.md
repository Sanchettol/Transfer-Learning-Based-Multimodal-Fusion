# Transfer-Learning-Based-Multimodal-Fusion
A Transfer Learning-Based Multimodal Emergency Detection System uses AI to detect accidents by combining image and text inputs. It applies ResNet18 for visual data and BERT for text analysis. Features are fused and classified to predict emergencies with confidence scores, accessible via a Flask-based web interface for real-time use.

# 🚨 Multimodal Emergency Detection System

## 📌 Overview

This project is a **Transfer Learning-Based Multimodal AI System** that detects emergency situations (like road accidents) using both **image and text inputs**.

It combines:

* 🖼️ Image Analysis (ResNet18)
* 📝 Text Analysis (BERT)
* 🔗 Feature Fusion (Multimodal Learning)

---

## 🎯 Features

* Upload an image (accident / normal scene)
* Enter a text message (e.g., "accident happened")
* AI predicts:

  * 🚨 Accident Detected
  * ✅ Normal Scene
* Displays **confidence score**
* Simple and interactive **Flask web interface**

---

## 🧠 Technologies Used

* Python
* PyTorch
* Transformers (HuggingFace BERT)
* Torchvision (ResNet18)
* Flask (Web Framework)
* HTML + Bootstrap (UI)

---

## ⚙️ Model Architecture

```
Image → ResNet18 → Image Features
Text → BERT → Text Features

        ↓
   Feature Fusion (Concatenation)

        ↓
 Fully Connected Layers

        ↓
 Prediction (Accident / Normal)
```

---

## 📂 Project Structure

```
project/
│
├── app.py
├── model.py
├── multimodal_model.pth
│
├── templates/
│   └── index.html
│
├── static/
│   └── uploads/
```

---
## Dataset
https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage

## 🚀 How to Run

### 1. Clone the repository

```
git clone <your-repo-link>
cd project
```

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install flask torch torchvision transformers pillow
```

### 4. Run the app

```
python app.py
```

### 5. Open in browser

```
http://127.0.0.1:5000/
```

---

## 📊 Output Example

```
🚨 Accident Detected (91.87%)
```

---

## 🧠 Key Concepts

* Transfer Learning
* Multimodal Learning
* Deep Learning (CNN + NLP)
* Feature Fusion

---

## 🎯 Future Enhancements

* Add Fire Detection 🔥
* Real-time CCTV integration 🎥
* Deploy on cloud 🌐
* Improve UI/UX 🎨

---

## 👨‍💻 Author

Sanchet

---




<img width="1912" height="1013" alt="image" src="https://github.com/user-attachments/assets/bec3c843-f7a7-409a-9b82-441e98497a8c" />

