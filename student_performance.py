# Student Performance Prediction
# Data Science Project using Linear Regression

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# --------------------------------------------------
# 1. Create Dataset
# --------------------------------------------------

data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "Attendance": [60, 65, 70, 72, 75, 80, 82, 85, 90, 95],
    "Previous_Score": [50, 55, 60, 62, 65, 70, 72, 78, 85, 88],
    "Final_Score": [52, 57, 63, 65, 68, 73, 75, 81, 88, 92]
}

df = pd.DataFrame(data)


# Display dataset
print("Student Performance Dataset:")
print(df)


# --------------------------------------------------
# 2. Separate Features and Target
# --------------------------------------------------

X = df[["Study_Hours", "Attendance", "Previous_Score"]]
y = df["Final_Score"]


# --------------------------------------------------
# 3. Split Dataset into Training and Testing Data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 4. Create and Train Linear Regression Model
# --------------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)


# --------------------------------------------------
# 5. Make Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


print("\nActual Scores:")
print(y_test.values)

print("\nPredicted Scores:")
print(y_pred)


# --------------------------------------------------
# 6. Evaluate Model
# --------------------------------------------------

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)


# --------------------------------------------------
# 7. Visualize Actual vs Predicted Scores
# --------------------------------------------------

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.title("Actual vs Predicted Student Scores")

plt.show()