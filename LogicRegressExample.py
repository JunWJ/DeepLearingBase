import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 读取数据
dataSet = pd.read_csv("./resources/breast_cancer_data.csv")
# print(dataSet)

# 提取x和y
x = dataSet.iloc[:, : -1]
y = dataSet['target']
# print(x)
# print(y)

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

# 数据归一化
sc = MinMaxScaler(feature_range=(0,1))
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


if __name__ == "__main__":
    # 逻辑回归模型训练
    logicRegress = LogisticRegression()
    logicRegress.fit(x_train, y_train)

    # 打印模型参数
    # print('w', logicRegress.coef_)
    # print('b', logicRegress.intercept_)

    # 模型评估
    y_pred = logicRegress.predict(x_test)
    # print('预测结果：', y_pred)

    # 打印预测结果的概率
    y_pred_proba = logicRegress.predict_proba(x_test)
    # print('预测概率：', y_pred_proba)

    # 得恶性肿瘤的概率
    pre_list = y_pred_proba[:, 1]
    # print('恶性肿瘤的概率：', pre_list)

    threshould = 0.3
    res = []
    res_name = []
    for i in range(len(pre_list)):
        if pre_list[i] >= threshould:
            res.append(1)
            res_name.append('恶性')
        else:
            res.append(0)
            res_name.append('良性')
    # print('结果：', res)
    # print('名称：', res_name)

    #模型评估指标
    report = classification_report(y_test, y_pred, labels=[0, 1], target_names=['良性', '恶性'])
    print(report)