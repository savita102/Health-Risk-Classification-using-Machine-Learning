🧬 **NovaGen Research Labs – Machine Learning Project**

📌 **Overview**

This project demonstrates an end-to-end machine learning workflow for a health risk classification problem, focusing on clean preprocessing, reproducible pipelines, and fair model comparison.
The goal is to build a predictive system that classifies individuals as “healthy” or “unhealthy” using clinical and lifestyle data while following industry-aligned best practices.

🎯 **Objective**

Train and compare multiple classification models

Apply proper preprocessing and scaling

Detect and control overfitting

Perform hyperparameter tuning

Select the best-performing model based on test data performance

🛠️ **Tech Stack**

Language: Python

Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

Techniques: Pipelines, Feature Scaling, Model Evaluation, Hyperparameter Tuning

🔍 **Workflow**

Data Understanding & EDA

Explored data structure, feature types, and statistical summaries

Preprocessing

Train-test split

Feature scaling using StandardScaler

Pipeline implementation (especially for SVM) to prevent data leakage

Modeling

Logistic Regression

Decision Tree (with pruning)

Random Forest

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Naive Bayes

Evaluation

Accuracy, Precision, Recall, F1-score

Classification Report on unseen test data

Model comparison to ensure fair selection

📈 **Results**

The initial Decision Tree showed signs of overfitting, which was reduced using pruning techniques.

Random Forest improved stability through ensemble learning.

KNN performance improved after selecting the optimal K value.

Feature scaling significantly improved SVM performance.

**Model Performance (Approx.)**
| Model                | Accuracy        |
| -------------------- | --------------- |
| Logistic Regression  | ~82%            |
| Naive Bayes          | ~82%            |
| Pruned Decision Tree | ~83%            |
| Random Forest        | ~85%            |
| KNN (Tuned)          | ~88%            |
| **SVM (Tuned)**      | **~93% (Best)** |


Final Choice: Support Vector Machine (SVM)
Reason: After scaling and hyperparameter tuning (RBF kernel), SVM achieved the highest accuracy with balanced precision and recall, showing strong generalization and effective boundary separation.

✅ **Key Learnings**

Pipelines are essential for clean and reproducible ML workflows

Feature scaling greatly impacts distance-based and margin-based models

Decision Trees can overfit without pruning

Ensemble models improve model stability

Model selection should rely on multiple metrics, not accuracy alone

Hyperparameter tuning can drastically improve performance

🚀 **Future Improvements**

Cross-validation for more robust evaluation

Advanced hyperparameter optimization (GridSearchCV / RandomizedSearchCV)

Feature importance and interpretability analysis

Model deployment (Streamlit / Flask)

👩‍💻 **Author**

Savita Pal
Aspiring Data Scientist | Python | Machine Learning | Data Analysis
