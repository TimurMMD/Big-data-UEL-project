# Big Data Analytics Project

## Overview
This repository contains the group project work for the Big Data Analytics module (CN7022). The project explores predictive capabilities of machine learning algorithms in forecasting term deposit subscriptions within the banking sector. Using a publicly available dataset from Kaggle, we implemented and compared multiple machine learning models to identify the most effective model for this classification task.

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Features](#key-features)
- [Setup Instructions](#setup-instructions)
- [Results and Insights](#results-and-insights)
- [Contributors](#contributors)

## Project Structure
```
├── UEL_Big_Data_Analytics_Assessment.docx    # Detailed project report
├── XGBoost.ipynb                             # Python notebook implementing the XGBoost model
├── README.md                                 # Project description (this file)
```

## Dataset
- **Source**: [Kaggle - Bank Marketing Campaign Dataset](https://www.kaggle.com/datasets/yaminh/bank-marketing-campaign-dataset/data)
- **Size**: 39.12 MB
- **Columns**: 11 (e.g., occupation, age, education level, call duration, etc.)
- **Rows**: 45,211
- **Goal**: Predict whether a customer will subscribe to a term deposit.

## Methodology
The project follows a structured approach combining the Scrum framework and KDD (Knowledge Discovery in Databases) methodology:

1. **Data Exploration**: Visualizations to understand feature distributions and relationships.
2. **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling.
3. **Model Training**: 
   - **Algorithms**: Random Forest, XGBoost, Logistic Regression, Decision Tree
   - Tools: PySpark, Python libraries (scikit-learn, XGBoost)
4. **Model Evaluation**: Metrics used include accuracy, precision, recall, and F1-score.
5. **Hyperparameter Tuning**: Optimization to improve model performance.

## Key Features
- **Machine Learning Models**: Comparative study of Random Forest, XGBoost, Logistic Regression, and Decision Tree.
- **Exploratory Data Analysis (EDA)**: Key insights on feature importance and relationships.
- **Evaluation Metrics**: ROC-AUC curves, precision-recall analysis, and confusion matrices.

## Setup Instructions
### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Libraries: PySpark, scikit-learn, XGBoost, Matplotlib, Pandas, NumPy

### Steps
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the repository:
   ```bash
   cd <repository_folder>
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Open and run the notebook:
   ```bash
   jupyter notebook XGBoost.ipynb
   ```

## Results and Insights
- **Best Performing Model**: XGBoost achieved the highest accuracy (90.34%) and F1-score (89.79%) among the tested algorithms.
- **Key Findings**:
  - Call duration strongly influences conversion rates.
  - Education level and previous campaign success significantly impact the likelihood of subscription.
  - Longer calls tend to result in higher conversions.

## Contributors
- **Low Yan Tong, Glenda**: Data cleaning, Random Forest, EDA
- **Santillan, Retxed Joshua**: Logistic Regression, Executive Summary
- **Timur Mamadaliyev**: XGBoost implementation, Cross-validation, Recommendations
- **Zhang Tai**: Decision Tree, EDA

---
