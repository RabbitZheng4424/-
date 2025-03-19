# =============== 数据预处理部分 ===============
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import math
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"E:\72h\diabetes.csv")

# 处理0值
cols_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_to_clean] = data[cols_to_clean].replace(0, np.nan)

# 填充缺失值
data.fillna(data.mean(), inplace=True)

# 删除重复值
data.drop_duplicates(inplace=True)

# =============== 数据拆分 ===============
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# 先拆分再缩放（防止数据泄露）
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =============== 特征缩放 ===============
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============== 模型训练部分 ===============
# 创建模型
model = RandomForestClassifier(
    n_estimators=100,
    max_features=int(math.sqrt(len(X.columns))),
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\n🎯 测试集准确率：{accuracy_score(y_test, y_pred)*100:.1f}%")

# =============== 预测展示部分 ===============
print("请输入以下信息来进行糖尿病预测：")

# 定义获取浮点数输入的函数
def get_float_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("输入无效，请输入一个数字。")

pregnancies = int(get_float_input("怀孕次数："))

def validate_glucose(value):
    if value < 20 or value > 300:  # 血糖合理范围
        raise ValueError("血糖值应在20-300之间")
    return value

glucose = validate_glucose(get_float_input("血糖："))
blood_pressure = get_float_input("血压：")
skin_thickness = get_float_input("皮肤厚度：")
insulin = get_float_input("胰岛素：")
bmi = get_float_input("BMI：")
diabetes_pedigree = get_float_input("糖尿病遗传函数：")
age = int(get_float_input("年龄："))

# 将输入的数据整理成模型需要的格式
new_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]]
new_data_df = pd.DataFrame(new_data, columns=data.columns[:-1])

# 缩放新数据
new_data_scaled = scaler.transform(new_data_df)

# 进行预测
prediction = model.predict(new_data_scaled)
probability = model.predict_proba(new_data_scaled)[0][1]  # 获取患病概率

# 输出每棵树的预测结果
tree_predictions = []
for i, tree in enumerate(model.estimators_, 1):
    pred = tree.predict(new_data_scaled)
    tree_predictions.append(pred[0])
    print(f"🌳 第{i}位树精灵的低语 → {'⚠️有风险' if pred[0]==1 else '🍃安全'}")

# 统计所有树的预测结果
final_vote = Counter(tree_predictions).most_common(1)[0][0]

# 输出最终预测结果
print(f"\n🌸 森林的温柔裁决：")
print(f"💖 患病概率: {probability*100:.1f}%")
print("❤️🩹 建议：" + ("请及时就医检查！" if prediction[0]==1 else "继续保持健康作息～"))

# =============== 统一打印特征重要性 ===============
print(f"\n🌳 每棵树使用特征数：{model.max_features}")
importance = pd.Series(model.feature_importances_, index=data.columns[:-1])
print("\n🌟 特征重要性排行榜：")
print(importance.sort_values(ascending=False).to_string())

# =============== 可视化预测分布 ===============
plt.figure(figsize=(15,6))

# 🌸 子图1：特征重要性星空图
plt.subplot(1,2,1)
importance = model.feature_importances_
features = data.columns[:-1]
colors = plt.cm.viridis(np.linspace(0,1,len(features)))  # 彩虹色系

bars = plt.barh(range(len(features)), importance, color=colors)
plt.yticks(range(len(features)), features)
plt.title('✨ 特征重要性星空图', fontsize=12, color='#FF69B4')
plt.xlabel('重要度', fontsize=10)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# 添加星形标记
for bar in bars:
    plt.scatter(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                marker='*', color='gold', s=100)

# 🌸 子图2：预测概率银河分布
plt.subplot(1,2,2)
probabilities = model.predict_proba(X_test)[:,1]  # 获取测试集的预测概率

# 绘制双峰分布
sns.kdeplot(probabilities[y_test==0], label='健康群体', fill=True, color='#2ecc71')
sns.kdeplot(probabilities[y_test==1], label='风险群体', fill=True, color='#e74c3c')

plt.title('🌌 预测概率银河分布', fontsize=12, color='#3498db')
plt.xlabel('患病概率', fontsize=10)
plt.ylabel('密度', fontsize=10)
plt.legend()
plt.grid(linestyle='--', alpha=0.5)

# 添加装饰元素
plt.annotate('低风险区域', xy=(0.2, 3), xytext=(0.1, 4),
             arrowprops=dict(facecolor='green', shrink=0.05),
             fontsize=9, color='darkgreen')
plt.annotate('高风险区域', xy=(0.8, 3), xytext=(0.7, 4),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=9, color='darkred')

plt.tight_layout()
plt.show()