import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
# 导入网格搜索法的函数
from sklearn.model_selection import GridSearchCV
# 导入GBDT模型的类
from sklearn.ensemble import GradientBoostingClassifier
#显示所有列
pd.set_option('display.max_columns', None)

# 数据读取
income = pd.read_excel(r'income_data.xlsx')
print(income.info())
# 查看数据集是否存在缺失值并使用pandas中fillna()方法，能够填充NA/NaN值
print(income.apply(lambda x: np.sum(x.isnull())))
income.fillna(value={'workclass': income['workclass'].mode()[0],
                     'occupation': income['occupation'].mode()[0],
                     'native-country': income['native-country'].mode()[0]}, inplace=True)
print(income.head())  # 输出前五行

# 设置绘图风格
plt.style.use('ggplot')
# 设置多图形的组合
fig, axes = plt.subplots(3, 1)
# 绘制不同收入水平下的受教育时长核密度图(针对数值型)
income['education-num'][income.income == ' >50K'].plot(kind='kde', label='>50K', ax=axes[0], legend=True, linestyle='-')
income['education-num'][income.income == ' <=50K'].plot(kind='kde', label='<=50K', ax=axes[0], legend=True,
                                                        linestyle='--')
# 绘制不同收入水平下的年龄核密度图
income.age[income.income == ' >50K'].plot(kind = 'kde', label = '>50K', ax = axes[1], legend = True, linestyle = '-')
income.age[income.income == ' <=50K'].plot(kind = 'kde', label = '<=50K', ax = axes[1], legend = True, linestyle = '--')


# 绘制不同收入水平下的周工作小时数密度图
income['hours-per-week'][income.income == ' >50K'].plot(kind='kde', label='>50K', ax=axes[2], legend=True,
                                                        linestyle='-')
income['hours-per-week'][income.income == ' <=50K'].plot(kind='kde', label='<=50K', ax=axes[2], legend=True,
                                                         linestyle='--')
# 显示图形
plt.savefig('plot/en_age_hpw.png', dpi=300)

# 构造不同收入水平下各种族人数的数据  (针对离散型)
race = pd.DataFrame(income.groupby(by=['race', 'income']).aggregate(np.size).loc[:, 'age'])
race = race.reset_index()  # 重设行索引
race.rename(columns={'age': 'counts'}, inplace=True)  # 变量重命名
race.sort_values(by=['race', 'counts'], ascending=False, inplace=True)  # 排序

# 构造不同收入水平下各家庭关系人数的数据
relationship = pd.DataFrame(income.groupby(by=['relationship', 'income']).aggregate(np.size).loc[:, 'age'])
relationship = relationship.reset_index()
relationship.rename(columns={'age': 'counts'}, inplace=True)
relationship.sort_values(by=['relationship', 'counts'], ascending=False, inplace=True)

# 设置图框比例，并绘图
plt.figure(figsize=(9, 5))
sns.barplot(x="race", y="counts", hue='income', data=race)
plt.savefig('plot/race_income.png', dpi=300)
plt.figure(figsize=(9, 5))
sns.barplot(x="relationship", y="counts", hue='income', data=relationship)
plt.savefig('plot/relat_income.png', dpi=300)

# 离散变量的重编码[workclass, education, marital-status,occupation, relationship, race, sex, native-country,income ]
for feature in income.columns:
    if income[feature].dtype == 'object':
        income[feature] = pd.Categorical(income[feature]).codes
# 删除变量(受教育程度和序号)
income.drop(['education', 'fnlwgt'], axis=1, inplace=True)
print(income.head())
print("离散变量重编码删除无关变量后: \n{}".format(income.head()))

# 数据集拆分
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(income.loc[:, 'age':'native-country'],
                                                    income['income'], train_size=0.75,
                                                    random_state=1234)
print('训练数据集共有%d条' % X_train.shape[0])
print('测试数据集共有%d条' % X_test.shape[0])

# 构建GBDT模型
gbdt = GradientBoostingClassifier()
gbdt.fit(X_train, y_train)
print(gbdt)

# 预测测试集
gbdt_pred = gbdt.predict(X_test)
print(pd.crosstab(gbdt_pred, y_test))

# 模型得分
print('模型在训练集上的准确率%f' %gbdt.score(X_train,y_train))
print('模型在测试集上的准确率%f' %gbdt.score(X_test,y_test))

# 绘制ROC曲线
plt.figure(figsize=(9,5))
fpr, tpr, _ = metrics.roc_curve(y_test, gbdt.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, linestyle = 'solid', color = 'red')
plt.stackplot(fpr, tpr, color = 'y')
plt.plot([0,1],[0,1], linestyle = 'dashed', color = 'black')
plt.text(0.6,0.4,'AUC=%.3f' % metrics.auc(fpr,tpr), fontdict = dict(size = 18))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.savefig('plot/GBDT_ROC.png',dpi=300)


# GBDT模型的网格搜索法
# 选择不同的参数（这里选择学习率和决策树最大深度进行调参）
learning_rate_options = [0.01, 0.05, 0.1]
max_depth_options = [3, 5, 7]
parameters = {'learning_rate': learning_rate_options, 'max_depth': max_depth_options}
grid_gbdt = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=parameters, cv=5, scoring='accuracy',n_jobs=-1)
grid_gbdt.fit(X_train, y_train)

# 结果输出
print("cv_results: {} \n best_params: {}\n best_score: {}".
      format(grid_gbdt.cv_results_, grid_gbdt.best_params_, grid_gbdt.best_score_))

# 预测测试集
grid_gbdt_pred = grid_gbdt.predict(X_test)
print(pd.crosstab(grid_gbdt_pred, y_test))

# 模型得分
print('模型在训练集上的准确率%f' % grid_gbdt.score(X_train, y_train))
print('模型在测试集上的准确率%f' % grid_gbdt.score(X_test, y_test))

# 绘制ROC曲线
plt.figure(figsize=(9,5))
fpr, tpr, _ = metrics.roc_curve(y_test, grid_gbdt.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, linestyle='solid', color='red')
plt.stackplot(fpr, tpr, color='steelblue')
plt.plot([0, 1], [0, 1], linestyle='dashed', color='black')
plt.text(0.6, 0.4, 'AUC=%.3f' % metrics.auc(fpr, tpr), fontdict=dict(size=18))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.savefig('plot/GBDTgrid_ROC.png', dpi=300)
