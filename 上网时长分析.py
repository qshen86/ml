#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as pl
mac2id=dict()
onlinetimes=[]
f=open('C:\data\TestData.txt','r', encoding='UTF-8')
for line in f:
	mac=line.split(',')[2]                                          #读取每条数据中的MAC地址，以,为分割，数组第三个值#
	onlinetime=int(line.split(',')[6])                              #读取上网时间，以,为分割，数组第7个值#
	starttime=int(line.split(',')[4].split(' ')[1].split(':')[0])   #读取开始时间，做了连续分割#
	if mac not in mac2id:                                           #如果mac不在mac2id这个字典中#
	    mac2id[mac]=len(onlinetimes)                                #将oneline值的长度赋值给该字典对应的Key-mac,也就是0/1/2...#
	    onlinetimes.append((starttime,onlinetime))                  #对onlinetimes数组赋值#
	else:
	   onlinetimes[mac2id[mac]]=[(starttime,onlinetime)]            #如果mac在mac2id这个字典中，对onlinetimes数组进行赋值#
real_X=np.array(onlinetimes).reshape((-1,2))                        #创建数组，改变数据形状，变成多维数组 [[*,*],[*,*]]#

print(mac2id)
print(onlinetimes)
print(real_X)

X=real_X[:,0:1]                                                     #将取starttime为新的数组#
print (X)
db=skc.DBSCAN(eps=0.01,min_samples=20).fit(X)                       #调用DBSCAN方法进行训练，labels为每个数据的簇标签#
labels = db.labels_

print('Labels:')                                                    #打印数据被标记的标签，计算标签为-1，即噪声数据的比例#
print(labels)
raito=len(labels[labels[:]==-1])/len(labels)
print('Noise raito:',format(raito,'.2%'))
n_clusters_=len(set(labels))-( 1 if -1 in labels else 0)            #set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据.计算簇的个数并打印，评价聚类效果#

print('Estimate number of clusters: %d' % n_clusters_)
print ("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X,labels))

for i in range(n_clusters_):                                        #打印各簇标号以及各簇内数据#
    print('Cluster ',i, ':')
    print (list(X[labels == i].flatten()))                          #如果labels == i为真则返回当前X的数值#

    
import matplotlib.pyplot as plt 
plt.hist(X,24)
plt.show()




