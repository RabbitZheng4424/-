
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
data = pd.read_csv(r"E:\72h\phase1_data\processed_diabetes.csv")
print("数据前5行：")
print(data.head())
print("\n数据信息：")
print(data.info())
print("\n数据描述：")
print(data.describe())

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=22, test_size=0.1)

transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

estimator = KNeighborsClassifier(n_neighbors=5)
estimator.fit(x_train, y_train)


y_pred = estimator.predict(x_test)
print("预测值是:\n", y_pred)
print("预测值和真实值对比:\n", y_pred == y_test)

score = estimator.score(x_test, y_test)
print("准确率是：", score)



def realtime_predict(model, scaler):
    print("\n❤️ 请输入患者数据（用空格分隔以下特征）：")
    print("怀孕次数 血糖 血压 皮肤厚度 胰岛素 BMI 糖尿病遗传函数 年龄")

    while True:
        try:

            input_str = input("\n请输入数据（输入 q 退出）: ").strip()
            if input_str.lower() == 'q':
                break


            data = list(map(float, input_str.split()))


            if len(data) != 8:
                raise ValueError("需要输入8个特征值")

            sample = scaler.transform([data])

            pred = model.predict(sample)[0]
            proba = model.predict_proba(sample)[0][1]

            print(f"\n🔮 预测结果：{'有糖尿病风险' if pred == 1 else '低风险'}")
            print(f"🩺 患病概率：{proba * 100:.1f}%")
            print("❤️🩹 建议：" + ("请及时就医检查！" if pred == 1 else "继续保持健康生活～"))

        except ValueError as ve:
            print(f"⚠️ 输入格式错误：{ve}")
        except Exception as e:
            print(f"❌ 发生异常：{str(e)}")


if __name__ == "__main__":

    print("\n" + "=" * 40)
    print("模型训练完成，进入实时预测模式")
    print("=" * 40)

    realtime_predict(estimator, transfer)