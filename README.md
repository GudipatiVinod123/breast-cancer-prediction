# 🩺 Breast Cancer Prediction using Logistic Regression

A Machine Learning web application that predicts whether a tumor is **Malignant** or **Benign** using Logistic Regression.

---
## Live link :
https://breast-cancer-prediction-hpcbvmp3rag3uefs4cxpj8.streamlit.app/

---
## 📌 Project Overview

This project uses:

- Logistic Regression Model
- Data Preprocessing with StandardScaler
- Flask Web Application
- Trained Model saved as `.pkl` file

The model is trained on breast cancer dataset and deployed using Flask.

---

## 📂 Project Structure

├── app.py
├── breast_cancer_model.pkl
├── scaler.pkl
├── data.csv
├── Breast_cancer_prediction_logistic_regression.ipynb
├── requirements.txt
└── README.md


---

## ⚙️ Technologies Used

- Python
- Scikit-Learn
- Pandas
- NumPy
- Flask
- Jupyter Notebook

---

## 🧠 Machine Learning Model

- Algorithm: Logistic Regression
- Model File: `breast_cancer_model.pkl`
- Scaler File: `scaler.pkl`

The model predicts:
- 0 → Malignant
- 1 → Benign

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Flask App
python app.py
4️⃣ Open in Browser
https://breast-cancer-prediction-hpcbvmp3rag3uefs4cxpj8.streamlit.app/
📊 Dataset
The dataset contains features like:

Radius

Texture

Perimeter

Area

Smoothness

Compactness

Symmetry

Fractal Dimension

📈 Model Training
Model training process is available in:

Breast_cancer_prediction_logistic_regression.ipynb
📌 Requirements
All dependencies are listed in:

requirements.txt
✨ Future Improvements
Deploy on Render / Railway

Add frontend UI improvements

Add model accuracy display

Add prediction probability score

👨‍💻 Author
Gudipati Vinod
Data Scientist | Machine Learning Enthusiast
