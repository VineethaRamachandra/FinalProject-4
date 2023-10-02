# FinalProject-4

Following are the files defined under FinalProject-4:

1. CreditCardFraudDetection.ipynb -> main source code file.
2. Fraud_Detection_for_Credit_Card_Transactions_Project4_Team4.docx  -> Initial project proposal document.
3. Credit_ard_Fraud_Detection-Project4-Group-4.pptx
4. creditcard.csv --> (PLEASE NOTE, this file was downloaded from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud. As the file size is around ~150MB, I was not able to upload the file due to Github restriction. Github does not allow one to upload file greater than 25 MB). For the sake of uploading the file, I have uploaded smaller set of data. However, for creating a model, I have used the ~ original ~150 MB csv file as is.
5. README>md file  -> Contains brief write up of this project, different steps, classification reprot, team member and contributions.
   
NOTE: The directory is clean and contains no extra files. We also do not have any API keys, so did not include a .gitignore file.

PLEASE NOTE: We cannot calculate the R-squared value, as R-squared is mostly applicable for Regression and our case is an example of pure binary classification.


   Project Team members and Contribution:
   1.	Andi Mysllinj
   2.	Krishna  Patel
   3.	Nancy Sakyi
   4.	Surender Raman
   5.	Susan Abraham
   6.	Vineetha Ramachandra: Developed end-to-end code, prepared Project proposal, updated README file, prepared project presentation deck.

Different steps involved in optimizing, creating the model and evaluating the model performance:


1. Downloaded the dataset from Kaggle and loaded it into preferred data analysis environment (e.g., Python with libraries like pandas, scikit-learn, and Matplotlib/Seaborn for data exploration).
   Also, created a Spark session to handle data processing, modeling, etc.
   
2. Explored the dataset to understand its structure, features, and class distribution. Noticed columns like "Time," "Amount," and "Class" (where "Class" is 0 for legitimate transactions and 1 for fraudulent transactions).
   
DATA OPTIMIZATION:

3. Handle Missing Values: Checked for any missing values in the dataset. If present, decide whether to impute or remove rows with missing values. In many cases, credit card transaction datasets have already been preprocessed and don't contain missing values.

4. Outlier Handling: Credit card fraud detection often involves handling outliers. Considered IQR to identify and potentially remove outliers in the "Amount" and "Time" columns.

5. Feature Scaling: Normalized the "Amount" and "Time" columns so that they have similar scales. Standard scaling (mean=0, std=1) is a common choice.

6. Class Imbalance Handling: Checked the class distribution to see if it's highly imbalanced. In most cases, found that the majority of transactions are legitimate (Class 0), while only a small percentage are fraudulent (Class 1).

7. Address Class Imbalance: To handle class imbalance, employed following technique:
	• Resampling: Either oversample the minority class (fraudulent transactions) or undersample the majority class (legitimate transactions) to balance the dataset.

8. Data Splitting: Split the preprocessed dataset into training and testing sets. A common split ratio is 70% for training and 30% for testing.

9. Model Selection: Chose a machine learning model suitable for binary classification, such as Random Forest or XGBoost. These models are robust and work well for fraud detection tasks.

10. Model Evaluation: Evaluated the model's performance on the testing dataset using appropriate metrics for fraud detection, including: - Precision: TP / (TP + FP) - Recall: TP / (TP + FN) - F1-Score: 2 * (Precision * Recall) / (Precision + Recall) - ROC AUC: Area under the Receiver Operating Characteristic curve.

11. Plotted the ROC curve and calculated the AUC score to assess the model's ability to distinguish between legitimate and fraudulent transactions.

CLASSIFICATION REPORT:

The "Classification Report" provides a summary of the performance metrics for a binary classification model. Here's what each of the metrics in the report means:

A. Precision (Positive Predictive Value): Precision measures the accuracy of positive predictions made by the model. In this context, it tells you how many of the predicted fraudulent transactions were actually fraudulent. A precision of 0.92 for Class 1 means that out of all transactions predicted as fraudulent, 92% were truly fraudulent.

B. Recall (Sensitivity or True Positive Rate): Recall measures the ability of the model to correctly identify all positive instances in the dataset. In this context, it tells us how many of the actual fraudulent transactions were correctly predicted by the model. A recall of 0.75 for Class 1 means that the model correctly identified 75% of all fraudulent transactions.

C. F1-Score: The F1-score is the harmonic mean of precision and recall. It is a single metric that balances precision and recall. A higher F1-score indicates a better balance between precision and recall. An F1-score of 0.83 for Class 1 suggests a reasonably good balance.

D. Support: Support represents the number of samples in each class. In ther report, Class 0 has 75,742 samples, and Class 1 has 129 samples. It provides context about the distribution of classes in the dataset.

E. Accuracy: Accuracy is the overall correctness of the model's predictions. It measures the ratio of correctly predicted samples to the total number of samples. An accuracy of 1.00 means that the model correctly classified all samples. However, in highly imbalanced datasets like fraud detection, accuracy can be misleading because it can be high even if the model doesn't perform well on the minority class (Class 1).

F. Macro Avg: This row provides the average of precision, recall, and F1-score calculated for both classes. It's useful for understanding the overall performance of the model without considering class imbalance.

G. Weighted Avg: Weighted average considers the number of samples in each class when calculating the average precision, recall, and F1-score. It gives more weight to the class with more samples, which is useful in imbalanced datasets.

H. ROC AUC Score: The ROC AUC (Receiver Operating Characteristic Area Under the Curve) score measures the model's ability to distinguish between the two classes across different probability thresholds. A score of 0.88 suggests that the model has reasonably good discriminatory power.

CONCLUSION: In summary, this classification report indicates that the model performs well in terms of precision, recall, and F1-score, especially for Class 0 (legitimate transactions). However, for Class 1 (fraudulent transactions), there is room for improvement in recall, which means that the model could better identify more fraudulent transactions while maintaining a high level of precision.



