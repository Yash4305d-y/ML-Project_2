# ❤️ Heart Disease Prediction Engine

An end-to-end machine learning project that predicts the likelihood of heart disease using multiple classification algorithms and custom feature engineering.

> **✅ Project Status:** Completed. Multiple machine learning models were trained, evaluated, and compared to identify the best-performing classifier.

---

## 🌐 Live Demo

🚀 Try the deployed application here:

**❤️ HeartWise ML:** https://heartwise-ml.streamlit.app/

The deployed web application uses the **K-Nearest Neighbors (KNN)** model, which achieved the highest performance during model evaluation.

---

## 📊 Preprocessing Pipeline

To ensure high-quality model performance and prevent data leakage, the dataset follows a structured preprocessing pipeline:

1. **Feature Engineering**

   * Created a custom `HR_BP_Ratio` feature (`MaxHR / RestingBP`)
   * Generated interaction features:

     * `MaxHR × Oldpeak`
     * `Age × Cholesterol`
     * `Oldpeak²`

2. **Categorical Encoding**

   * Applied One-Hot Encoding using `drop_first=True` to avoid multicollinearity.

3. **Feature Scaling**

   * Standardized numerical features using `StandardScaler` for algorithms sensitive to feature magnitude.

4. **Model Training**

   * Trained and evaluated multiple classification algorithms using the engineered feature set.

---

## 💡 Core Architecture

Rather than relying solely on raw medical attributes, this project enhances predictive performance through custom feature engineering and interaction terms. These engineered features help the models capture complex relationships between patient characteristics that may not be evident from individual variables alone.

---

## 📊 Model Performance Comparison

| Rank | Model                         |  Accuracy  |  F1-Score  |
| :--: | ----------------------------- | :--------: | :--------: |
|  🥇  | **K-Nearest Neighbors (KNN)** | **89.67%** | **0.9073** |
|  🥈  | Logistic Regression           | **87.50%** | **0.8856** |
|  🥉  | Naive Bayes                   | **87.50%** | **0.8844** |
|  4️⃣ | Support Vector Machine (SVM)  | **85.33%** | **0.8670** |
|  5️⃣ | Decision Tree                 | **75.54%** | **0.7805** |

---

## 🏆 Best Performing Model

| Metric   | Model                         |      Score |
| -------- | ----------------------------- | ---------: |
| Accuracy | **K-Nearest Neighbors (KNN)** | **89.67%** |
| F1-Score | **K-Nearest Neighbors (KNN)** | **0.9073** |

**Analysis:**
Among all evaluated classifiers, **K-Nearest Neighbors (KNN)** achieved the highest overall performance, delivering the best balance between prediction accuracy and F1-score. Logistic Regression and Naive Bayes also performed strongly, while Decision Tree showed comparatively lower generalization on the test set.

---

## 🧪 Models Evaluated

* ✅ Logistic Regression
* ✅ K-Nearest Neighbors (KNN)
* ✅ Support Vector Machine (SVM)
* ✅ Naive Bayes
* ✅ Decision Tree

---

## 🛠️ Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit

---

## 🔮 Future Improvements

* Hyperparameter tuning using GridSearchCV and RandomizedSearchCV
* Cross-validation for robust model evaluation
* Feature selection and dimensionality reduction
* Ensemble learning (Random Forest, XGBoost)
* Enhance the Streamlit application with prediction history and richer visualizations

---

## 📌 Conclusion

This project demonstrates a complete machine learning workflow for heart disease prediction, including data preprocessing, feature engineering, model training, evaluation, comparison, and deployment.

Five classification algorithms were evaluated, with **K-Nearest Neighbors (KNN)** emerging as the best-performing model, achieving an **Accuracy of 89.67%** and an **F1-Score of 0.9073**. The trained KNN model has been deployed as an interactive Streamlit web application for real-time heart disease risk prediction.

The project highlights how thoughtful preprocessing and feature engineering can significantly improve predictive performance on structured medical datasets.

---

⭐ If you found this project useful, consider giving it a star!
