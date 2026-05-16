# 🏭 Air Quality Impact on Workplace Productivity

A machine learning project predicting worker productivity from 
environmental sensor data — directly aligned with AirSynQ Systems' 
IoT air quality monitoring platform.
Built as part of my Data Science learning journey.

---

## 📌 Project Overview

This project investigates whether real-time air quality sensor 
readings can predict workplace productivity levels. Rather than 
simply classifying air quality as Good, Moderate or Unhealthy, 
the model engineers a continuous Productivity Score from pollutant 
concentrations and uses it as a regression target.

The core business question: can an IoT air quality system predict 
worker output decline before it happens?

---

## 📂 Dataset

- **Source:** UCI Machine Learning Repository — Air Quality Dataset
- **Coverage:** Hourly sensor readings from an Italian city
- **Size:** 9,358 hourly observations
- **Sensors:** CO, NOx, NO2, O3, NMHC, Benzene, Temperature, Humidity

---

## 🛠️ What Was Done

**Data Collection and Cleaning**
Loaded the UCI Air Quality dataset with semicolon separation and 
European decimal formatting. Handled missing values encoded as -200. 
Converted date and time columns to proper datetime format.

**Feature Engineering**
Engineered 24 features from raw sensor readings including:

- CO 1hr, 2hr and 3hr rolling averages and lag features
- Rolling PM2.5 average and Pollution Spike flag
- Synthetic PM2.5 derived from NOx and CO sensor readings
- Temperature deviation from daily mean
- Humidity deviation from daily mean
- Hour of day for time-based patterns
- Absolute Humidity calculated from temperature and relative humidity

**Target Variable — Productivity Score**
A synthetic Productivity Score was engineered from environmental 
readings based on occupational health research:

High CO reduces oxygen delivery to the brain.
High NO2 impairs cognitive function.
High Benzene causes fatigue and neurological effects.
Temperature extremes reduce physical and mental performance.

The score represents estimated worker productivity as a percentage 
relative to baseline clean air conditions.

**Models Trained**
Four regression models were trained and compared using 
chronological time-based splitting.

**Backtesting and Forecast Curves**
Time-series realignment chart comparing all model forecast curves 
against actual productivity baseline across the test period.

---

## 📊 Model Results

| Model | R² Score | RMSE |
|---|---|---|
| Linear Regression | 0.589 | 5.46% |
| Random Forest | 0.538 | 5.79% |
| CatBoost | 0.512 | 5.95% |
| XGBoost | 0.468 | 6.21% |

Linear Regression achieved the strongest performance — suggesting 
that the relationship between pollutant concentrations and 
productivity follows relatively linear patterns at this scale.

---

## 🔑 Feature Importance

Top drivers of productivity prediction across all models:

| Rank | Feature | Significance |
|---|---|---|
| 1 | NO2(GT) | Primary cognitive impairment pollutant |
| 2 | Synthetic PM2.5 | Respiratory impact on output |
| 3 | Relative Humidity | Thermal comfort and fatigue |
| 4 | NOx(GT) | Combustion byproduct exposure |
| 5 | PT08.S2(NMHC) | Hydrocarbon sensor signal |

NO2 was the strongest predictor across Random Forest and XGBoost 
at 28% importance — consistent with occupational health research 
on nitrogen dioxide and cognitive performance.

---

## 💡 Key Findings

Air quality sensor readings explain 59% of workplace productivity 
variance using Linear Regression. NO2 and PM2.5 are the strongest 
environmental predictors of productivity decline. Rolling averages 
and lag features improved model performance by capturing cumulative 
exposure effects. An IoT deployment of this model could flag 
productivity risk in real time before workers notice symptoms.

---

## 🏢 Business Relevance

This project directly maps to AirSynQ Systems' core product offering — 
continuous IoT air quality monitoring with predictive alerts for 
mines, offices, hotels and industrial facilities.

The model demonstrates that sensor data alone can quantify 
productivity impact, providing organisations with a financial 
justification for investing in air quality monitoring systems.

---

## 🧰 Tools & Libraries

- Python 3
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- CatBoost
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 📁 File Structure
Air-Quality-Productivity/
│
├── Notebook.ipynb              # Main analysis notebook
├── AirQualityUCI.csv           # Raw dataset
└── README.md                   # Project documentation

---

## 👤 Author

**Lindokuhle Nyoka**
Aspiring ML Engineer — Incoming Electrical & Electronic Engineer 
at AirSynQ Systems
[GitHub](https://github.com/lk-nyoka) · 
[LinkedIn](https://linkedin.com/in/lindokuhle-nyoka-982019245)