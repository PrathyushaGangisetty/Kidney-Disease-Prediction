# Kidney-Disease-Prediction

🧠 Chronic Kidney Disease (CKD) Prediction Using Machine Learning and SMOTE
This project focuses on building a predictive model for Chronic Kidney Disease (CKD) using a combination of machine learning algorithms and oversampling techniques. It was implemented in both Python (using libraries like Scikit-Learn, Keras, TensorFlow) and IBM SPSS Modeler for comparative performance evaluation.

Project Overview
Chronic Kidney Disease is a persistent condition affecting kidney function. Early detection is critical for effective treatment. This project uses classification algorithms such as SVM, Random Forest, Logistic Regression, and Deep Neural Networks (ANN) to detect CKD with high accuracy, while handling data imbalance using SMOTE (Synthetic Minority Oversampling Technique).

🎯 Objectives
Predict the presence of CKD from patient medical data.

Handle class imbalance using SMOTE.

Evaluate and compare multiple classification algorithms.

Implement both traditional ML models and ANN.

Perform feature selection to optimize accuracy.


⚙️ Technologies Used
Python: NumPy, Pandas, Matplotlib, Scikit-Learn, Keras, TensorFlow

IBM SPSS Modeler: For visual ML modeling and performance benchmarking

ML Techniques: SVM, Logistic Regression, Random Forest, ANN

Data Handling: KNN Imputation, Label Encoding, Standard Scaler

Oversampling: SMOTE (from imbalanced-learn)


🧪 Process Flow (IBM SPSS Modeler)
Data ingestion and type assignment

Partitioning for train-test split

SMOTE oversampling on minority class

Model training: LSVM, C5.0, CHAID, Random Forest, Logistic Regression

Evaluation on accuracy, AUC, lift, and profit metrics

📊 Best Accuracy:

LSVM with SMOTE: 98.53% accuracy, 0.999 AUC

Deep Neural Network: Comparable high performance with optimized training

📈 Model Evaluation Metrics
Accuracy

Area Under Curve (AUC)

Precision, Recall, F1 Score

Confusion Matrix

Lift & Profit Charts (SPSS)

