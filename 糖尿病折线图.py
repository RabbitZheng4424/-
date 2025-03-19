import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 数据集应包含以下8个特征列：
df = pd.read_csv(r"E:\72h\diabetes.csv")
features = ['Pregnancies', 'Glucose', 'BloodPressure',
           'SkinThickness', 'Insulin', 'BMI',
           'DiabetesPedigreeFunction', 'Age']

# 2. 数据预处理
scaler = MinMaxScaler()  # 创建归一化器（将所有值压缩到0-1之间）
scaled_data = scaler.fit_transform(df[features])  # 自动计算并转换数据

# 3. 创建画布
plt.figure(figsize=(12, 6), dpi=100)  # 设置画布尺寸和分辨率

# 4. 绘制前20个患者的折线图（避免图像太密集）
for i in range(20):
    plt.plot(features,  # X轴：特征名称
            scaled_data[i],  # Y轴：归一化后的特征值
            marker='o',  # 每个数据点显示圆圈
            alpha=0.5,  # 设置半透明看清重叠线
            linewidth=1)  # 线条粗细

# 5. 添加图表元素
plt.title('糖尿病患者特征分布折线图', fontsize=14, pad=20)
plt.xlabel('生理特征', fontsize=12)
plt.ylabel('归一化值', fontsize=12)
plt.xticks(rotation=45)  # X轴标签旋转45度防重叠

# 6. 显示图表
plt.tight_layout()  # 自动调整元素间距
plt.show()