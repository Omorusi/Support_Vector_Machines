# Spam Classification using SVM 
This project classifies text messages as either spam or ham (not spam) for feature extraction and Support Vector Machines (SVM) for classification.

## Project Overview
Load and preprocess the dataset
Convert text messages into numerical features using TF-IDF
Apply Principal Component Analysis (PCA) to reduce feature dimensions
Train a Support Vector Machine (SVM) model on the dataset
Visualize decision boundaries using matplotlib and seaborn
Installation
Ensure you have the necessary libraries installed:


Category: ham (0) or spam (1)
Message: The text content of the message
The Category column is mapped as follows:

python
Copy
Edit
data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})
Code Workflow
## 1. Load Data
python
Copy
Edit
import pandas as pd

![image alt](https://github.com/Omorusi/Support_Vector_Machines/blob/main/Screenshot%202025-03-03%20155532.png?raw=true)

 ## 2. Feature Extraction (TF-IDF)
### Changing Columns names 
![image alt](https://github.com/user-attachments/assets/d10b9c84-91e7-413f-b597-f71b1301bf7e)
![image alt](https://github.com/Omorusi/Support_Vector_Machines/blob/main/Screenshot%202025-03-03%20155548.png?raw=true)

 ## 3. Train-Test Split
python
Copy
Edit
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
4. Dimensionality Reduction (PCA)
python
Copy
Edit
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
5. Train SVM Model
python
Copy
Edit
from sklearn.svm import SVC

svm_model = SVC(kernel='linear')
svm_model.fit(X_train_pca, y_train)
6. Visualize Decision Boundary
python
Copy
Edit
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
A custom function is used to plot decision boundaries.

Results & Evaluation
Accuracy is calculated using:
python
Copy
Edit
from sklearn.metrics import accuracy_score

y_pred = svm_model.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
Visualization shows the separation between spam and non-spam messages.
Conclusion
TF-IDF + SVM is effective for spam detection.
PCA reduces dimensionality while retaining classification performance.
The decision boundary visualization provides insight into model behavior.
