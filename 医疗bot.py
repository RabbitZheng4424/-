import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv(r"E:\72h\diabetes.csv")
print("原始数据前5行:\n", data.head())

print("缺失值情况:\n", data.isnull().sum())

cols_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_to_clean] = data[cols_to_clean].replace(0, np.nan)

print("处理0值后的缺失值情况:\n", data.isnull().sum())

data.fillna(data.mean(), inplace=True)

data.drop_duplicates(inplace=True)

scaler = MinMaxScaler()
data[cols_to_clean] = scaler.fit_transform(data[cols_to_clean])

data.to_csv("processed_diabetes.csv", index=False)
print("数据处理完成，已保存为 processed_diabetes.csv")

glucose_normalized = data['Glucose'].values
print("归一化后的前5个Glucose值:", glucose_normalized[:5])