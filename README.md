🔍 **Project Overview**

NovaGen Research Labs conducts large-scale population health studies to understand how underlying health conditions influence disease risk and long-term outcomes. This project develops a predictive machine learning model to classify individuals as “healthy” or “unhealthy” using physiological measurements, lifestyle factors, and medical history attributes.

This classification aids in:

Selecting eligible participants for clinical trials and longitudinal studies

Stratifying populations for risk-based analysis and outcome comparison

The study includes:

Baseline models

Tree-based models

Ensemble learning

Distance-based learning

Margin-based learning

Probabilistic learning

🎯 **Objectives**

Implement multiple classification algorithms

Evaluate models using proper metrics

Detect and handle overfitting

Perform hyperparameter tuning

Compare models and select the best performer

🤖 **Algorithms Used**
| Algorithm                    | Purpose                  |
| ---------------------------- | ------------------------ |
| Logistic Regression          | Linear baseline model    |
| Decision Tree                | Tree-based classifier    |
| Pruned Decision Tree         | Overfitting control      |
| Random Forest                | Ensemble learning        |
| Support Vector Machine (SVM) | Margin-based classifier  |
| K-Nearest Neighbors (KNN)    | Distance-based model     |
| Naive Bayes                  | Probabilistic classifier |

⚙️ **Preprocessing Steps**

Data cleaning

Train-test split

Feature scaling (StandardScaler)

Pipeline implementation (for SVM)

Hyperparameter tuning (K for KNN, tree pruning, SVM parameters)

📊 **Evaluation Metrics**

The following metrics were used for fair model comparison:

Accuracy

Precision

Recall

F1-Score

Classification Report

These metrics ensure evaluation beyond accuracy, especially for class balance.

🚨 **Overfitting Analysis**

The initial Decision Tree achieved 100% training accuracy, indicating overfitting.

Pruning techniques (max_depth, min_samples_leaf) reduced variance and improved generalization.

Random Forest further reduced overfitting using ensemble learning.

🔧 **Model Tuning Highlights**
| Model         | Improvement Technique                   |
| ------------- | --------------------------------------- |
| Decision Tree | Pruning                                 |
| Random Forest | Ensemble averaging                      |
| KNN           | Optimal K selection (best at K=9)       |
| SVM           | Feature scaling + RBF kernel + pipeline |

🏆 **Model Performance Comparison**
| Model                | Accuracy        |
| -------------------- | --------------- |
| Logistic Regression  | ~82%            |
| Naive Bayes          | ~82%            |
| Pruned Decision Tree | ~83%            |
| Random Forest        | ~85%            |
| KNN (K=9)            | ~88%            |
| **SVM (Tuned)**      | **~93% (Best)** |

🧠 **Key Learnings**

Accuracy alone is not enough; precision, recall, and F1-score are crucial

Decision Trees easily overfit without pruning

Ensemble models improve stability

Scaling is essential for distance-based and margin-based models

Hyperparameter tuning can drastically change model performance

Model selection depends on data characteristics

🥇 **Final Conclusion**

After testing multiple models and applying tuning techniques, the SVM classifier with feature scaling and RBF kernel achieved the best performance (~93% accuracy) with balanced precision and recall. This indicates strong generalization and effective learning of the decision boundary.

🛠️ **Technologies Used**

Python

Scikit-learn

Pandas

NumPy
