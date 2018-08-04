import pandas as pd
import seaborn as s
import matplotlib.pyplot as plt


#%% Get and Clean Data#%% Get

#Read data as pandas dataframe
d = pd.read_csv('diagnostic.csv')
df = d.drop('Unnamed: 32', axis=1)

#if using diagnosis as categorical
df.diagnosis = df.diagnosis.astype('category')

#Create references to subset predictor and outcome variables
x = list(df.drop('diagnosis',axis=1).drop('id', axis=1))
y = 'diagnosis'

#show first 10 rows
print df.head(10)

#Explore correlations#Explor
plt.rcParams['figure.figsize'] = (15, 10)
s.set(font_scale=1.4)
s.heatmap(df.drop('diagnosis', axis=1).drop('id', axis=1).corr(), cmap='coolwarm')

#plt.show()
plt.ion()
plt.pause(3)
plt.close()


plt.rcParams['figure.figsize'] = (16, 9)
f, axes = plt.subplots(2, 5)
s.boxplot('diagnosis',    y='radius_mean',             data=df,     ax=axes[0, 0])
s.boxplot('diagnosis',    y='texture_mean',            data=df,     ax=axes[0, 1])
s.boxplot('diagnosis',    y='perimeter_mean',          data=df,     ax=axes[0, 2])
s.boxplot('diagnosis',    y='area_mean',               data=df,     ax=axes[0, 3])
s.boxplot('diagnosis',    y='smoothness_mean',         data=df,     ax=axes[0, 4])
s.boxplot('diagnosis',    y='compactness_mean',        data=df,     ax=axes[1, 0])
s.boxplot('diagnosis',    y='concavity_mean',          data=df,     ax=axes[1, 1])
s.boxplot('diagnosis',    y='concave points_mean',     data=df,     ax=axes[1, 2])
s.boxplot('diagnosis',    y='symmetry_mean',           data=df,     ax=axes[1, 3])
s.boxplot('diagnosis',    y='fractal_dimension_mean',  data=df,     ax=axes[1, 4])
f.tight_layout()
plt.ion()
plt.pause(3)
plt.close()


plt.rcParams['figure.figsize'] = (16, 9)
f, axes = plt.subplots(2, 5)
s.boxplot('diagnosis',    y='radius_se',               data=df,    ax=axes[0, 0],    palette='cubehelix')
s.boxplot('diagnosis',    y='texture_se',              data=df,    ax=axes[0, 1],    palette='cubehelix')
s.boxplot('diagnosis',    y='perimeter_se',            data=df,    ax=axes[0, 2],    palette='cubehelix')
s.boxplot('diagnosis',    y='area_se',                 data=df,    ax=axes[0, 3],    palette='cubehelix')
s.boxplot('diagnosis',    y='smoothness_se',           data=df,    ax=axes[0, 4],    palette='cubehelix')
s.boxplot('diagnosis',    y='compactness_se',          data=df,    ax=axes[1, 0],    palette='cubehelix')
s.boxplot('diagnosis',    y='concavity_se',            data=df,    ax=axes[1, 1],    palette='cubehelix')
s.boxplot('diagnosis',    y='concave points_se',       data=df,    ax=axes[1, 2],    palette='cubehelix')
s.boxplot('diagnosis',    y='symmetry_se',             data=df,    ax=axes[1, 3],    palette='cubehelix')
s.boxplot('diagnosis',    y='fractal_dimension_se',    data=df,    ax=axes[1, 4],    palette='cubehelix')
f.tight_layout()
plt.ion()
plt.pause(3)
plt.close()


plt.rcParams['figure.figsize']=(16, 9)
f, axes = plt.subplots(2, 5)
s.boxplot('diagnosis',    y='radius_worst',              data=df,    ax=axes[0, 0],    palette='coolwarm')
s.boxplot('diagnosis',    y='texture_worst',             data=df,    ax=axes[0, 1],    palette='coolwarm')
s.boxplot('diagnosis',    y='perimeter_worst',           data=df,    ax=axes[0, 2],    palette='coolwarm')
s.boxplot('diagnosis',    y='area_worst',                data=df,    ax=axes[0, 3],    palette='coolwarm')
s.boxplot('diagnosis',    y='smoothness_worst',          data=df,    ax=axes[0, 4],    palette='coolwarm')
s.boxplot('diagnosis',    y='compactness_worst',         data=df,    ax=axes[1, 0],    palette='coolwarm')
s.boxplot('diagnosis',    y='concavity_worst',           data=df,    ax=axes[1, 1],    palette='coolwarm')
s.boxplot('diagnosis',    y='concave points_worst',      data=df,    ax=axes[1, 2],    palette='coolwarm')
s.boxplot('diagnosis',    y='symmetry_worst',            data=df,    ax=axes[1, 3],    palette='coolwarm')
s.boxplot('diagnosis',    y='fractal_dimension_worst',   data=df,    ax=axes[1, 4],    palette='coolwarm')
f.tight_layout()
plt.ion()
plt.pause(3)
plt.close()


