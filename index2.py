# SMART IRRIGATION SYSTEM
# PREDICT: WATER / NO WATER
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. LOAD DATA
data = pd.read_csv("dataset2.csv")

print(data.head())

# 2. CREATE TARGET COLUMN (1 > need water, 0 > no water)
def irrigation_logic(row):
    if row["SoilMoisture"] < 300:
        return 1 
    elif row["SoilMoisture"] < 500 and row["temperature"] > 30:
        return 1
    elif row["Humidity"] < 30:
        return 1
    else:
        return 0 

data["IrrigationNeeded"] = data.apply(irrigation_logic, axis=1)

print("\nTarget distribution:")
print(data["IrrigationNeeded"].value_counts())

# 3. FEATURES & TARGET
X = data[["CropDays", "SoilMoisture", "temperature", "Humidity"]]
y = data["IrrigationNeeded"]

# 4. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. TRAIN MODEL
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 6. PREDICTIONS
y_pred = model.predict(X_test)

# 7. ACCURACY
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# 8. CONFUSION MATRIX
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - Irrigation System")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 9. REPORT
print(classification_report(y_test, y_pred))

# 10. FEATURE IMPORTANCE
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(6,4))
plt.barh(features, importances)
plt.title("Feature Importance")
plt.show()

# 11. PREDICTION FUNCTION
def predict_irrigation(crop_days, soil_moisture, temp, humidity):
    input_data = np.array([[crop_days, soil_moisture, temp, humidity]])
    pred = model.predict(input_data)[0]

    if pred == 1:
        return "WATER REQUIRED"
    else:
        return "NO WATER NEEDED"

# 12. TEST
print(predict_irrigation(10, 200, 35, 20))
print(predict_irrigation(10, 600, 20, 35))
