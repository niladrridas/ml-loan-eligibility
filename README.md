# 🚀 Loan Eligibility Prediction

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-3776AB?style=flat-square&logo=python&logoColor=white)
![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white)

Welcome to the **Loan Eligibility Prediction** project, where machine learning is leveraged to predict loan eligibility based on the financial and demographic information of applicants. The dataset, sourced from Kaggle, includes a variety of features used to train a logistic regression model for accurate predictions.

## 💾 Dataset Overview

The dataset used in this project contains several key features:

- **Gender**
- **Marital Status**
- **Education**
- **Applicant's Income**
- **Co-applicant's Income**
- **Loan Amount**
- **Loan Term**
- **Credit History**
- **Loan Status** (Target Variable)

The dataset can be found [here](https://www.kaggle.com/datasets/zhijinzhai/loandata) on Kaggle.

## 🛠️ Tech Stack

This project is built using Python and several powerful libraries commonly used in machine learning:

- **[Python](https://www.python.org/)**
- **[NumPy](https://numpy.org/)** - Numerical computing
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation and analysis
- **[scikit-learn](https://scikit-learn.org/)** - Machine learning model building and evaluation
- **[Matplotlib](https://matplotlib.org/)** - Data visualization

## 🧑‍💻 Methodology

This project follows a structured approach, covering the entire machine learning pipeline:

1. **Data Import**: The dataset is imported using `pandas.read_csv()`.
2. **Data Preprocessing**: Missing values are handled, categorical variables are encoded, and feature scaling is applied to ensure the model performs optimally.
3. **Model Training**: A logistic regression model is trained using `LogisticRegression` from `scikit-learn`.
4. **Model Evaluation**: Performance is measured using accuracy, precision, recall, and F1-score.
5. **Prediction**: Loan eligibility predictions are made using the trained model.

## 📊 Model Performance

### 🔥 **Accuracy**: 1.0

### 📈 **Classification Report**:

```
                    precision    recall  f1-score   support

        COLLECTION       1.00      1.00      1.00        24
COLLECTION_PAIDOFF       1.00      1.00      1.00        24
           PAIDOFF       1.00      1.00      1.00        52

          accuracy                           1.00       100
         macro avg       1.00      1.00      1.00       100
      weighted avg       1.00      1.00      1.00       100
```

### 📊 **Confusion Matrix**:

```
[[24  0  0]
 [ 0 24  0]
 [ 0  0 52]]
```

## 📄 Report

🔗 [View Report](/doc/report.pdf)

## 💻 Usage

### Follow these steps to run the project:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/niladrridas/ml-loan-eligibility.git
   cd ml-loan-eligibility
   ```

2. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Main Script**:
   ```bash
   python main.py
   ```

4. **View Results and Predictions** directly in the terminal or through visual outputs.

## 📚 References

- **Dataset**: [Kaggle - Loan Eligibility Dataset](https://www.kaggle.com/datasets/zhijinzhai/loandata)
- **scikit-learn documentation**: [scikit-learn](https://scikit-learn.org/stable/)
- **pandas documentation**: [pandas](https://pandas.pydata.org/docs/)

---

Feel free to fork this repository, submit issues, or contribute with improvements. Happy coding! 🎉