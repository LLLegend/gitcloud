import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
df_qingdao = pd.read_csv('air_qingdao_2018.csv', header=None, names=["Date", "Quality_grade", "AQI", "AQI_rank", "PM2.5"])
# 读取文件
df_qingdao.dropna(axis=0)   # 删除缺省值
fig = plt.figure(figsize=(10, 10), dpi=100)  # 调整画布大小
ax1 = fig.add_subplot(221)
ax1.plot(df_qingdao['Date'].values, df_qingdao['AQI'].values, color='red')
ax1.set_xlabel(" 日期 ")   # 横坐标标签
ax1.set_ylabel(" AQI指数 ")  # 纵坐标标签
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='both',alpha=0.4)
ax2 = fig.add_subplot(222)
data_bar = list(df_qingdao["Quality_grade"])
result = {}
for i in set(data_bar):
    result[i] = data_bar.count(i)    # 计算出每个天气状况对应的数量，以字典形式存储
keys = list(result.keys())
values = list(result.values())
ax2.bar(keys, values, color='rbg')
for a, b in result.items():
    plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)   # 柱状图上方显示数量
ax3 = fig.add_subplot(223)
data_pie = list(df_qingdao["AQI_rank"])
result = {"r<50": 0, "50<=r<100": 0, "100<=r<150": 0, "150<=r<200": 0, "200<=r<250": 0, "250<=r<300": 0, "r>=300": 0}
for i in data_pie:
    if i < 50:
        result["r<50"] = result["r<50"] + 1
    if 50 <= i < 100:
        result["50<=r<100"] = result["50<=r<100"] + 1
    if 100 <= i < 150:
        result["100<=r<150"] = result["100<=r<150"] + 1
    if 150 <= i < 200:
        result["150<=r<200"] = result["150<=r<200"] + 1
    if 200 <= i < 250:
        result["200<=r<250"] = result["200<=r<250"] + 1
    if 250 <= i < 300:
        result["250<=r<300"] = result["250<=r<300"] + 1
    if i >= 300:
        result["r>=300"] = result["r>=300"] + 1
keys = list(result.keys())
values = list(result.values())
ax3.pie(values, labels=keys, autopct='%1.2f%%')  # 小数点后两位
ax4 = fig.add_subplot(224)
ax4.plot(df_qingdao['Date'].values, df_qingdao['PM2.5'].values)
plt.savefig(r'E:\python\天气爬虫与数据分析\pic.png')  # 保存图片
plt.show()   # 显示图片
pm = np.array(df_qingdao["PM2.5"]).reshape(-1, 1)
aqi = np.array(df_qingdao["AQI"]).reshape(-1, 1)
print("两数据之间的相关性系数为:", np.corrcoef(pm.reshape(1, -1), aqi.reshape(1, -1)))   # 相关性系数,
plt.scatter(pm, aqi)   # 传入矩阵,散点图
# 将原始数据通过随机方式分割为训练集和测试集，其中测试集占比为40%
x_train, x_test, y_train, y_test = train_test_split(pm, aqi, test_size=0.1)  # 将数据分为训练集和测试集，测试集占10%
lrModel = LinearRegression()
lrModel.fit(x_train, y_train)
# 模型训练
'''
调用模型的fit方法，对模型进行训练
这个训练过程就是参数求解的过程
并对模型进行拟合
'''
print("对回归模型的评分", lrModel.score(x_test, y_test))   # 对回归模型进行检验
print("回归模型的截距为：", lrModel.intercept_)  # 查看截距
print("回归模型的回归系数为：", lrModel.coef_[0][0])     # 查看参数
plt.plot(pm, lrModel.predict(pm), color='blue')
print("数据与预测模型的平均差值为", math.sqrt(((y_test - lrModel.predict(x_test)) ** 2).sum())/32)
plt.show()









