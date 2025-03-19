# =============== 数据预处理部分 ===============
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import math

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
# 特征缩放
# =============== 特征缩放 ===============
scaler = MinMaxScaler()
X_train[cols_to_clean] = scaler.fit_transform(X_train[cols_to_clean])
X_test[cols_to_clean] = scaler.transform(X_test[cols_to_clean])
# =============== 模型训练部分 ===============
X = data.drop('Outcome', axis=1)
y = data['Outcome']
# 创建模型
model = RandomForestClassifier(
    n_estimators=100,
    max_features=int(math.sqrt(len(X.columns))),
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\n🎯 测试集准确率：{accuracy_score(y_test, y_pred)*100:.1f}%")
# 训练模型

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
new_data_df[cols_to_clean] = scaler.transform(new_data_df[cols_to_clean])
# 进行预测
prediction = model.predict(new_data_df)  # 注意用回model而不是tree
probability = model.predict_proba(new_data_df)[0][1]  # 获取患病概率

# 输出每棵树的预测结果
tree_predictions = []
for i, tree in enumerate(model.estimators_, 1):
    pred = tree.predict(new_data_df.values)
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
importance = pd.Series(model.feature_importances_, index=X.columns)
print("\n🌟 特征重要性排行榜：")
print(importance.sort_values(ascending=False).to_string())