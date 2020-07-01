import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['font.serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

df = pd.read_excel(r'RFM电商数据.xlsx')
# print(df.head())
df = df.loc[df['order_status'] == '交易成功', :]  # delete the Refundtrade
# print('剔除退款后还剩 %d 行' % len(df))
df = df[['nickname', 'payment_time', 'pay_amount']]  # extract key fields
# construct key fields
# step 1: calculate Recency
r = df.groupby('nickname')['payment_time'].max().reset_index()  # to get the last payment time of all
# print(r.head())
r['R'] = (pd.to_datetime('2020-6-1') - r['payment_time']).dt.days
r = r[['nickname', 'R']]
# print(r.head())
# step 2: calculate Frequency
df['payment_date'] = pd.to_datetime(df['payment_time'].apply(lambda x: x.date()))
df_tmp = df.groupby(['nickname', 'payment_date'])['payment_time'].count().reset_index()
f = df_tmp.groupby('nickname')['payment_time'].count().reset_index()
f.columns = ['nickname', 'F']
# print(f.head())
# step 3: calculate Monetary
sum_m = df.groupby('nickname')['pay_amount'].sum().reset_index()
sum_m.columns = ['nickname', 'total_pay_amount']
m = pd.merge(sum_m, f)
#m['M'] = m['total_pay_amount'] / m['F']
m['M'] = m['total_pay_amount']
# print(m.head())
# get RFM value of each user
rfm = pd.merge(r, m)
rfm = rfm[['nickname', 'R', 'F', 'M']]
# print(rfm.head())
rfm.to_csv('rfm.csv')

# View data distribution
plt.figure(figsize=(6, 4))
sns.set_style("darkgrid", {"font.sans-serif": 'SimHei'})
'''
sns.distplot(rfm['R'])
plt.title('Recency分布直方图',fontsize = 15)
plt.savefig('r.png')
plt.show()
sns.countplot(rfm['F'])
plt.title('Frequency分布直方图',fontsize = 15)
plt.savefig('f.png')
plt.show()
sns.distplot(rfm['M'],color = 'g')
plt.title('Monetary分布直方图',fontsize = 15)
plt.savefig('m.png')
plt.show()
'''
# Data processing and model construction
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics

# Data standardization
rfm_s = rfm.copy()
min_max_scaler = preprocessing.MinMaxScaler()
rfm_s = min_max_scaler.fit_transform(rfm[['R', 'F', 'M']])
inertia = []
ch_score = []
ss_score = []
'''
for k in range(2, 10):
    model = KMeans(n_clusters=k, init='k-means++', max_iter=500)
    model.fit(rfm_s)
    pre = model.predict(rfm_s)
    ch = metrics.calinski_harabasz_score(rfm_s, pre)
    ss = metrics.silhouette_score(rfm_s, pre)
    inertia.append(model.inertia_)
    ch_score.append(ch)
    ss_score.append(ss)
print(ch_score, ss_score, inertia)

score = pd.Series([ch_score, ss_score, inertia], index=['calinski_harabaz_score', 'silhouette_score', 'inertia'])
aa = score.index.tolist()
plt.figure(figsize=(15, 6))
j = 1
for i in aa:
    plt.subplot(1, 3, j)
    plt.plot(list(range(2, 10)), score[i])
    plt.xlabel('k值', fontsize=13)
    plt.title(f'{i}值', fontsize=15)
    j += 1
plt.subplots_adjust(wspace=0.3)
#plt.savefig('metrics.png')
plt.show()
# k = 6
'''
model = KMeans(n_clusters=6, init='k-means++', max_iter=500)
model.fit(rfm_s)
res = model.predict(rfm_s)
res = pd.DataFrame(res)
data = pd.concat([rfm, res], axis=1)
data.rename({0: u'cluster'}, axis=1, inplace=True)
print(data.head())  #Get the corresponding cluster of each user

# Get the center point of each category
labels = model.labels_
labels = pd.DataFrame(labels, columns=['category'])
result = pd.concat([pd.DataFrame(model.cluster_centers_), labels['category'].value_counts().sort_index()], axis=1)
result.columns = ['Recency', 'Frequence', 'Monetary', 'Number of samples in the cluster']
#result.to_csv('clusterResult.csv')
print(result)

for k in range(6):
    aa = ['R', 'F', 'M']
    plt.figure(figsize=(18, 4))  # 设置绘图区的大小：18 X 4
    j = 1
    for i in aa:
        plt.subplot(1, 3, j)  # 设置1行3列的图片区
        a = data[i]
        sns.kdeplot(data[data['cluster'] == k][i], color='r', shade=True)  # 第k个簇的数据分布
        plt.title(i, fontsize=14)
        j += 1
    #plt.savefig('cluster_'+str(k)+'.png')
    #plt.show()


