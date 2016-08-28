"""
Load data from PDSD

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



import sklearn as sk
import sklearn.neighbors
import sklearn.svm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

#-----------------------------------------------------------#
def get_sec(s):
    return np.asarray([int(l[0]) * 3600 + int(l[1]) * 60 + int(l[2]) for l in s])


#-----------------------------------------------------------#

with open('pdsd_2015.csv') as f:
    data=pd.read_csv(f, header=0)
#remove bad columns
data.drop(data.columns[[range(3,12)]],axis=1,inplace=True)
#remove bad rows
data=data.dropna()

###################################
###################################
#convert time, time of day
#for some reason, it's backwards
FullDate=pd.to_datetime(data.date_time,format='%Y-%m-%d %H:%M:%S')
TimeOfDay=get_sec(data.date_time.str.split(' ').str.get(1).str.split(':'))
DayOfWeek=data.day.values
# print type(DayOfWeek[1])
#Priority
Priority=data.priority.values

#Beat
Beat=data.beat.values

#see if beat, time of day can be used to predict priority

###################################
###################################
ips=np.column_stack((DayOfWeek, TimeOfDay, Beat))
ops=np.transpose(Priority)
print ops.size
skf = StratifiedKFold(ops, n_folds=5)

#guess maximum number
print('Gu2: Guess p=2, the most common priority')
print('KNN: K nearest neighbors')
print('SGD: Stochastic Gradient Descent')
print('KAP: Kernel Approximation')
print('RNF: Random Forrest')

for fold, (tr, te) in enumerate(skf):
    knn=sk.neighbors.KNeighborsClassifier(n_neighbors=4)
    sgd = SGDClassifier(loss="hinge", penalty="l2")
    rbf_feature = RBFSampler(gamma=1, random_state=1)
    X_features = rbf_feature.fit_transform(ips)
    kap=SGDClassifier()

    rnf = RandomForestClassifier(n_estimators=4, max_depth=None,
        min_samples_split=1, random_state=0)


    knn.fit(ips[tr],ops[tr])
    sgd.fit(ips[tr],ops[tr])
    kap.fit(ips[tr],ops[tr])
    rnf.fit(ips[tr],ops[tr])
    print('Fold:%i, Gu2:%.3f, KNN:%.3f, SGD:%.3f, KAP:%.3f, RNF: %.3f' %
        (fold, accuracy_score(ops[te],np.ones(te.size)*2),
         accuracy_score(ops[te], knn.predict(ips[te])),
         accuracy_score(ops[te], sgd.predict(ips[te])),
         accuracy_score(ops[te], kap.predict(ips[te])),
         accuracy_score(ops[te], rnf.predict(ips[te]))))
