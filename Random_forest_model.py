import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split as tts

df = pd.read_csv('RAT01.csv')
df1 = pd.read_csv('RAT02.csv')
df2 = pd.read_csv('RAT03.csv')
df3 = pd.read_csv('RAT04.csv')
df4 = pd.read_csv('RAT05.csv')
df5 = pd.read_csv('RAT06.csv')
df6 = pd.read_csv('RAT07.csv')
df7 = pd.read_csv('RAT08.csv')
for i in (df.columns.values):
    print(type(i))

df.head()
df.shape

dfs = []  # List to store the DataFrames

for i in range(1, 9):
    file_name = f'RAT0{i}.csv'  # Construct the file name
    df = pd.read_csv(file_name)  # Read the CSV file into a DataFrame
    df['Target'] = file_name[:-4]  # Add an 'Target' column with values as filename without extection
    dfs.append(df)  # Append the DataFrame to the list
df_merge = pd.concat(dfs,ignore_index=True)

print(df_merge.columns)
Y = df_merge['Target']
Y = Y.map({'RAT01':0,
           'RAT02':1,'RAT03':2,'RAT04':3,'RAT05':4,'RAT06':5,'RAT07':6,'RAT08':7})
X = df_merge.drop("Target",axis=1)
X.shape

X.Source.value_counts().sort_values(ascending=False).head(10)

top_10 = [x for x in X.Source.value_counts().sort_values(ascending=False).head(10).index]
top_10

for label in top_10:
    X[label]=np.where(X['Source']==label,1,0)
X[['Source']+top_10].head(5)


top_10 = [x for x in X.Destination.value_counts().sort_values(ascending=False).head(10).index]
top_10


for label in top_10:
    X[label]=np.where(X['Destination']==label,1,0)
X[['Destination']+top_10].head(5)



top_10 = [x for x in X.Protocol.value_counts().sort_values(ascending=False).head(10).index]
top_10


for label in top_10:
    X[label]=np.where(X['Protocol']==label,1,0)
X[['Protocol']+top_10].head(5)
X.drop(['Source','Destination','Protocol','Info'],axis=1,inplace=True)
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(X,Y,test_size=.3,random_state=1)
print(x_train.shape)
print(x_test.shape)
print(y_train.head())
X.head()
from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy='auto',random_state=42)
x_res,y_res = sm.fit_resample(x_train,y_train)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_res)
x_test = sc.transform(x_test)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=60 , max_depth=20)
clf.fit(x_train,y_res)
clf.score(x_train,y_res)
from sklearn.metrics import classification_report
y_pred_dt = clf.predict(x_test)
print(classification_report(y_test,y_pred_dt))

from sklearn.metrics import confusion_matrix
cmat = confusion_matrix(y_test,y_pred_dt) 
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=60 , max_depth=20)
clf.fit(x_train,y_res)

from sklearn import tree
for i in range(3):
    estimator = clf.estimators_[i]
    plt.figure(figsize=(10, 10))
    tree.plot_tree(estimator, filled=True)
    plt.show()


acc=clf.score(x_train,y_res)

acc = round(acc*100,2)
print(f'Accuracy of Random Forest classifier on training set: {acc} %')

from sklearn import tree
dec = tree.DecisionTreeClassifier(max_depth=8)
dec = dec.fit(x_train,y_res)
acc = dec.score(x_train,y_res)
plt.figure(figsize=(20,20))
tree.plot_tree(dec,filled=True)
plt.show()

print(f'Accuracy of Decision Tree classifier on training set: {round(acc*100,2)} %')
