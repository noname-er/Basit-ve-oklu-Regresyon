import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


diabetes = load_diabetes()

X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

X_bmi = X[["bmi"]]  # Tek bir özellik: BMI

X_bmi_train, X_bmi_test, y_train, y_test = train_test_split(X_bmi, y, test_size=0.2, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_bmi_train, y_train)

y_pred_simple = model_simple.predict(X_bmi_test)

r2_simple = r2_score(y_test, y_pred_simple)
mae_simple = mean_absolute_error(y_test, y_pred_simple)
mse_simple = mean_squared_error(y_test, y_pred_simple)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_multiple = LinearRegression()
model_multiple.fit(X_train, y_train)
y_pred_multiple = model_multiple.predict(X_test)

r2_multiple = r2_score(y_test, y_pred_multiple)
mae_multiple = mean_absolute_error(y_test, y_pred_multiple)
mse_multiple = mean_squared_error(y_test, y_pred_multiple)

print("Basit Lineer Regresyon (BMI):")
print(f"R²: {r2_simple:.3f}, MAE: {mae_simple:.2f}, MSE: {mse_simple:.2f}")

print("\nÇoklu Lineer Regresyon (Tüm Özellikler):")
print(f"R²: {r2_multiple:.3f}, MAE: {mae_multiple:.2f}, MSE: {mse_multiple:.2f}")


