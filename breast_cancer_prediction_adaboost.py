import numpy as np
import pandas as pd
import seaborn as s
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

d = pd.read_csv('diagnostic.csv')
df = d.drop('Unnamed: 32', axis=1)

#if using diagnosis as categorical
df.diagnosis = df.diagnosis.astype('category')

#Create references to subset predictor and outcome variables
x = list(df.drop('diagnosis',axis=1).drop('id', axis=1))
y = 'diagnosis'

np.random.seed(10)


traindf, testdf = train_test_split(df, test_size = 0.3)

x_train = traindf[x]
y_train = traindf[y]

x_test = testdf[x]
y_test = testdf[y]

dt=DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
ada_real = AdaBoostClassifier(base_estimator=dt, learning_rate=1,
                              n_estimators=1000, algorithm='SAMME.R')
AdaBoostfit = ada_real.fit(x_train, y_train)
accuracy = AdaBoostfit.score(x_test, y_test)
predict = AdaBoostfit.predict(x_test)
cmatrix = confusion_matrix(y_test, predict)

print ('Accuracy of Random Forest: %s' % "{0:.2%}".format(accuracy))

# Cross_Validation
v = cross_val_score(AdaBoostfit, x_train, y_train, cv=10)
for i in range(10):
    print('Cross Validation Score: %s' % '{0:.2%}'.format(v[i, ]))

ax = plt.axes()
s.heatmap(cmatrix, annot=True, fmt='d', ax=ax, cmap='BrBG', annot_kws={"size": 30})
ax.set_title('Random Forest Confusion Matrix')
plt.ion()
plt.pause(3)
plt.close()
