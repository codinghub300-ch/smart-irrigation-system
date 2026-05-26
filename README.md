# Smart Irrigation System 

An AI-powered Smart Irrigation System developed using Python, Machine Learning, and Streamlit to predict whether crops need watering based on soil and environmental conditions.

---

##  Features

- Predict irrigation requirements using AI
- Interactive Streamlit dashboard
- Real-time user input analysis
- Machine Learning model using Random Forest
- Confusion Matrix visualization
- Feature Importance analysis
- Dataset statistical overview
- Smart farming decision support

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn

---

##  Project Structure

smart-irrigation-system/
│
├── index.py
├── index2.py
├── dataset2.csv
└── README.md

---

##  Machine Learning Model

The system uses:

- Random Forest Classifier

Input Features:
- Crop Days
- Soil Moisture
- Temperature
- Humidity

Output:
- 1 → WATER REQUIRED
- 0 → NO WATER NEEDED

---

## Run Terminal Version
python index2.py


## Run Web Dashboard
streamlit run index.py


## Dashboard Features
User-friendly interface
Sidebar controls for environmental inputs
Real-time irrigation prediction
Performance visualization
Dataset analytics
AI explanation section


## Visualizations Included
Confusion Matrix
Feature Importance Graph
Dataset Statistics
Irrigation Distribution Chart

 
 ## Project Goals

This project aims to:
Improve water management in agriculture
Support smart farming solutions
Apply machine learning in real-world scenarios
Reduce unnecessary irrigation

##  How to Run

### 1. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
