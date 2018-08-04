import numpy as np
import pandas as pd
import seaborn as s
from xgboost import XGBClassifier
from xgboost import plot_importance

from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
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

model = XGBClassifier(learning_rate=0.3,
                      n_estimators=1000,
                      max_depth=6,
                      gamma=0)

model.fit(x_train, y_train)
y_pred_class = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred_class)
cmatrix = confusion_matrix(y_test, y_pred_class)
v = cross_val_score(model, x_train, y_train, cv=10)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
for i in range(10):
    print('Cross Validation Score: %s' % '{0:.2%}'.format(v[i, ]))
plot_importance(model)

plt.ion()
plt.pause(3)
plt.close()

ax = plt.axes()
s.heatmap(cmatrix, annot=True, fmt='d', ax=ax, cmap='BrBG', annot_kws={"size": 30})
ax.set_title('Random Forest Confusion Matrix')
plt.ion()
plt.pause(3)
plt.close()
