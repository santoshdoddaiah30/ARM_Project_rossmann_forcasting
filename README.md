# Rossmann Demand Forecasting (ARM Project)

This repository contains an end-to-end **consumer demand forecasting project** in the retail sector using the **Rossmann Store Sales dataset**.  
The project follows a full pipeline: **data preprocessing → exploratory data analysis (EDA) → feature engineering → model training → evaluation → final forecasting output**.

---

## Project Objective
To forecast daily retail sales (consumer demand) and analyze the impact of:
- Promotions
- Holidays
- Seasonality
- Store characteristics
- Competition-related features

---

## Repository Structure
- `scripts/` → Python scripts for each step of the pipeline  
- `figures/` → All generated plots (EDA + evaluation + feature importance)  
- `reports/` → Final project report (Word document)  
- `data/` → Contains final forecast output (`submission.csv`)  

---

## Dataset
Dataset used: **Rossmann Store Sales (Kaggle Competition)**  
https://www.kaggle.com/c/rossmann-store-sales

**Note:** The original dataset (`train.csv`, `test.csv`, etc.) is not uploaded here because GitHub website uploads have file-size limits.  
Please download the dataset from Kaggle and place it in a local `data/` folder when running the code.

---

## Models Implemented
- **Linear Regression** (Baseline)
- **Random Forest Regressor**
- **XGBoost Regressor (Final Model)**

---

## Final Model Performance (XGBoost Validation)
- **RMSE:** 925.72  
- **MAE:** 644.53  
- **R²:** 0.908  

---

## Output
- `data/submission.csv` → Final predicted sales output generated using the XGBoost model

---

## How to Run 

In Terminal: 

### 1) Clone Repository
```bash
git clone https://github.com/santoshdoddaiah30/ARM_Project_rossmann_forcasting.git
```

### 2) Change Directory
```bash
cd ARM_Project_rossmann_forcasting
```

### 3) Install Dependencies
```bash
pip install -r requirements.txt
```

### 4) Run the project file
```bash
python scripts/pipeline.py
```
### 5) Run Dashboard
```bash
streamlit run scripts/dashboard.py
```


