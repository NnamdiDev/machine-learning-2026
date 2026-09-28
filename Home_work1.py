import pandas as pd
import numpy as np

# print(pd.__version__)

df = pd.read_csv('car_fuel_efficiency_2026.csv')

# print(df.head())
# print(df.shape[0])
# print(df['fuel_type'].nunique())
# print(df.isna().sum())
# print(df[df['origin'] == 'Asia']['fuel_efficiency_mpg'].max())
# print(df['horsepower'].median())
# print(df['horsepower'].mode())
# df['horsepower'] = df['horsepower'].fillna(df['horsepower'].mode()[0])
# print(df['horsepower'].median())
X = df[df['origin'] == 'Asia'][['vehicle_weight', 'model_year']].head(7)

# print(X)
X = X.to_numpy()
# print(X)

XTX = X.T @ X 
# print(XTX)
XTX_inv = np.linalg.inv(XTX)

# print(XTX_inv)

y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = XTX_inv @ X.T @ y

print(w.sum())
# print(w)
# print(y)