# Bone-Fracture-Detection
## Introduction
 Since long ago, bone fractures was a long standing issue for mankind, and it's classification via x-ray has always depended on human diagnostics – which may be sometimes flawed.
In recent years, Machine learning and AI based solutions have become an integral part of our lives, in all aspects, as well as in the medical field.
In the scope of our research and project, we have been studying this issue of classification and have been trying, based on previous attempts and researches, to develop and fine tune a feasible solution for the medical field in terms of identification and classification of various bone fractures, using CNN ( Convolutional Neural Networks ) in the scope of modern models, such as ResNet, DenseNet, VGG16, and so forth.
After performing multiple model fine tuning attempts for various models, we have achieved classification results lower then the predefined threshold of confidence agreed upon later in this research, but with the promising results we did achieve, we believe that systems of this type, machine learning and deep learning based solutions for identification and classification of bone fractures, with further fine tuning and applications of more advanced techniques such as Feature Extraction, may replace the traditional methods currently employed in the medical field, with much better results.


## Dataset
The data set we used called MURA and included 3 different bone parts, MURA is a dataset of musculoskeletal radiographs and contains 20,335 images described below:


| **Part**     | **Normal** | **Fractured** | **Total** |
|--------------|:----------:|--------------:|----------:|
| **Elbow**    |    3160    |          2236 |      5396 |
| **Hand**     |    4330    |          1673 |      6003 |
| **Shoulder** |    4496    |          4440 |      8936 |

The data is separated into train and valid where each folder contains a folder of a patient and for each patient between 1-3 images for the same bone part

## ⚙️ Model Architecture

- **Backbone:** ResNet50 (pretrained on ImageNet, frozen)
- **Layers Added:**
  - Global Average Pooling
  - Dense (128 units, ReLU)
  - Dense (50 units, ReLU)
  - Output (3 units, Softmax)
- **Optimizer:** Adam (lr=0.0001)
- **Loss:** Categorical Crossentropy
- **Input Shape:** 224×224 RGB images

---

## 📊 Results

- **Test Accuracy:** ~75–80% (varies with tuning)
- **Plots:** Accuracy & Loss over training epochs
- **Early Stopping** prevents overfitting


## 🖥️ Streamlit Web App

The app allows:
- Uploading X-ray images
- Predicting the **bone type**
- (Optional) Predicting **fracture status**
- Displaying model confidence and charts

---

## 🐳 Docker & Cloud Deployment

### Build Docker Image

```bash
docker build -t gcr.io/YOUR_PROJECT_ID/bone-fracture-app .


