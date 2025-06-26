# 🔌 Electricity Bill Estimator using Machine Learning

Predict household electricity bills based on real-world survey and climate data using machine learning regression models.

---

## 📦 Project Overview

This project uses data from the Residential Energy Consumption Survey (RECS) to estimate annual electricity bill amounts. It applies Random Forest Regression to learn patterns from regional, housing, and climate-related features.

---

## 🧠 ML Model Used

- ✅ **Random Forest Regression**

---

## 📊 Features Used

| Feature       | Description                            |
|---------------|----------------------------------------|
| `REGIONC`     | Region code                            |
| `TYPEHUQ`     | Housing type                           |
| `CDD65`       | Cooling Degree Days                    |
| `HDD65`       | Heating Degree Days                    |
| `OA_LAT`      | Latitude of residence                  |
| `GWT`         | Ground water temperature               |
| `DesignDBT1`  | Design dry bulb temperature            |

---

## 🧪 Model Evaluation

- Visualized **Actual vs Predicted** results using matplotlib
- Handled missing values and scaled features

---

## 📁 Files

| File            | Purpose                             |
|-----------------|-------------------------------------|
| `main.py`       | ML pipeline and model training      |
| `elec_bill.csv` | Cleaned dataset                     |
| `requirements.txt` | Python dependencies             |

---

## 📈 Results

> ✅ Model trained with over 20,000+ samples  
> 📉 Achieved consistent prediction accuracy (visuals included)  
> 🚀 Ready for Streamlit dashboard deployment

---

## 🚀 Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main script
python main.py
