'''Poisson Regression is a type of regression used to model count data or rates, especially when the dependent variable represents counts or the number of occurrences of an event in a fixed interval of time or space. The response variable is assumed to follow a Poisson distribution, and it is typically used when the data exhibit non-negative integer values .'''

'''1. Install Necessary Libraries
You need the following libraries for this task:

pandas for data manipulation.
numpy for numerical computations.
statsmodels for Poisson regression.
matplotlib for visualization.'''

'pip install pandas numpy statsmodels matplotlib

'''2. Import Libraries and Load Dataset
For demonstration, we’ll use a simple dataset from statsmodels. In practice, you would load your own dataset. Let's assume a fictional dataset similar to bike-sharing data for Poisson regression.'''

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Let's create a fictional dataset
np.random.seed(42)

# Simulating data: 
# y (count of bike rentals), X (weather, temperature, humidity, time of day)
n_samples = 500
X1 = np.random.normal(25, 5, n_samples)  # Temperature
X2 = np.random.uniform(50, 100, n_samples)  # Humidity
X3 = np.random.randint(1, 5, n_samples)  # Time of day (1-4)
X4 = np.random.uniform(0, 1, n_samples)  # Weather (0 - good weather, 1 - bad weather)
# Lambda (rate parameter for Poisson distribution)
lambda_rate = np.exp(0.1 * X1 + 0.03 * X2 - 0.5 * X3 + 0.5 * X4)  # Non-linear relationship
y = np.random.poisson(lambda_rate)  # Generating the count data

# Create a DataFrame
df = pd.DataFrame({'Temperature': X1, 'Humidity': X2, 'Time_of_day': X3, 'Weather': X4, 'Rentals': y})

# Display the first few rows
df.head()

'''3. Split the Data into Training and Testing Sets
To evaluate the model, we will split the data into training and testing sets.'''

# Step 1: Split the data into training and testing sets
X = df[['Temperature', 'Humidity', 'Time_of_day', 'Weather']]  # Independent variables
y = df['Rentals']  # Dependent variable

# Train-Test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional: Standardize the features (important for some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


'''4. Fit the Poisson Regression Model
Now, we can fit the Poisson regression model using the Poisson class from statsmodels.'''

# Step 2: Add constant for the intercept
X_train_scaled = sm.add_constant(X_train_scaled)

# Step 3: Fit the Poisson regression model
poisson_model = sm.GLM(y_train, X_train_scaled, family=sm.families.Poisson()).fit()

# Step 4: Print the summary of the model
print(poisson_model.summary())

