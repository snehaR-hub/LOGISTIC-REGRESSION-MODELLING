'''Logistic Regression is a statistical method used for binary classification tasks. It is used when the dependent variable is binary (0/1, Yes/No, True/False). The model predicts the probability that a given input point belongs to a certain class (usually class 1).

We will demonstrate Logistic Regression using the Iris dataset for a binary classification task (predicting whether the Iris flower is of species "Setosa" or "Not Setosa"). We'll use scikit-learn to perform the logistic regression.'''

'''1. Import Libraries and Load the Dataset
We will use the Iris dataset for this demonstration. The dataset has 150 samples, with 4 features (sepal length, sepal width, petal length, petal width) and 3 classes (Setosa, Versicolor, Virginica). For binary classification, we will only use the first two classes .'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# We will perform binary classification, so select only "Setosa" vs "Not Setosa"
y_binary = (y == 0).astype(int)  # 1 if 'Setosa', 0 otherwise

# Use only the first two features (for simplicity and visualization)
X_binary = X[:, :2]

# Convert to DataFrame for ease of handling
X_df = pd.DataFrame(X_binary, columns=iris.feature_names[:2])

# Check the shape of the data
print("Features shape:", X_df.shape)
print("Target shape:", y_binary.shape)

'''2. Preprocess the Data
Before applying Logistic Regression, we should:

Split the data into training and testing sets.
Standardize the features to ensure they have a mean of 0 and a standard deviation of 1'''

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_df, y_binary, test_size=0.3, random_state=42)

# Step 3: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check the scaled features

print("Scaled features (first 5 rows of X_train):")
print(X_train_scaled[:5])

'''3. Train the Logistic Regression Model
Now that the data is preprocessed, we can train the Logistic Regression model using the LogisticRegression class from scikit-learn.'''

# Step 4: Train the Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

# Step 5: Make predictions on the test set
y_pred = logreg.predict(X_test_scaled)

'''4. Evaluate the Model
After training the model, we evaluate its performance using:

Accuracy: The proportion of correct predictions.
Confusion Matrix: Shows the true positives, true negatives, false positives, and false negatives.
Classification Report: Includes precision, recall, f1-score, and support for each class.'''

# Step 6: Evaluate the model's performance

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
cr = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(cr)

'''5. Visualizing the Results
We can visualize the decision boundary and the data points using a scatter plot, where different colors represent different classes.'''

# Step 7: Visualize the decision boundary
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1

# Create meshgrid for plotting decision boundary
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Predict class labels for the meshgrid
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and scatter plot of data points
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', s=100)
plt.title('Logistic Regression - Decision Boundary')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.colorbar(label='Class')
plt.show()
