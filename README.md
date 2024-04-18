# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
Developed By:Lokesh N
Reg no:212222100023
```
```
import pandas as pd 
from scipy import stats
import numpy as np

df=pd.read_csv("/content/bmi.csv")
df.head()
```
![Screenshot 2024-04-16 105854](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/16eab2bb-3f7a-4272-bfdd-7399873cc400)

```
df.dropna()
```
![Screenshot 2024-04-16 105909](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/c203fa73-f0a9-41c0-a34a-2d187a7d79eb)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![Screenshot 2024-04-16 105917](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/420f9b88-b9d5-4357-9156-1a79cb75be72)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![Screenshot 2024-04-16 105928](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/a0b85cee-87c9-45c0-ad30-b98979a72586)

```
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![Screenshot 2024-04-16 105940](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/c9fe4a0a-817a-49b6-9c40-9690fb32bc29)

```
from sklearn.preprocessing import Normalizer
Scaler=Normalizer
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
```
![Screenshot 2024-04-16 105951](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/b60db3cc-8937-41b4-b705-760a7b0b71c3)

```
df=pd.read_csv("/content/bmi.csv")
```
```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
```
![Screenshot 2024-04-16 110002](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/8c05bac6-2074-448d-a01c-957602f0a6cd)

```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![Screenshot 2024-04-16 110012](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/eae58633-258e-46ee-b2eb-1cc83a22f25f)

```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[" ?"])
data
```
![Screenshot 2024-04-16 110034](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/facec5ac-f677-4d8b-8671-26734cfd9aac)

```
data.isnull().sum()
```
![Screenshot 2024-04-16 110040](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/a5768ebf-66fe-4bca-a53e-cc5659bdc19f)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![Screenshot 2024-04-16 110101](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/f55fe384-827c-4beb-bc24-d0c507e0b3a5)

```
data2=data.dropna(axis=0)
data2
```
![Screenshot 2024-04-16 110113](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/70be9e57-2103-4bb9-a1bb-46c63b372604)

```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than  or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![Screenshot 2024-04-16 110137](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/55d8323d-49d4-46d9-98e4-5a4d7a441eb1)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![Screenshot 2024-04-16 110147](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/8573a802-4e29-4883-b08c-883ff9e9755e)

```
data2
```
![Screenshot 2024-04-16 110158](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/04ea419d-9017-449e-bc4b-8e9f12d34467)

```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```
![Screenshot 2024-04-16 110224](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/dddc2238-fe6f-42ed-a701-3f99271ba37a)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![Screenshot 2024-04-16 110243](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/ee2594cf-0704-432c-9c7d-128fc627cb10)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![Screenshot 2024-04-16 110300](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/0ab98965-d622-4f0c-b56c-c3e4a5688253)

```
y=new_data['SalStat'].values
print(y)
```
![Screenshot 2024-04-16 110306](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/a77e9962-ffcc-47ae-bc72-c257855f9a23)

```
x=new_data[features].values
print(x)
```
![Screenshot 2024-04-16 110312](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/8db68d75-b466-4d94-b9db-ed249b878b1f)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)
prediction=KNN_Classifier.predict(test_x)
confusionMmatrix=confusion_matrix(test_y,prediction)
print(confusionMmatrix)
```
![image](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/f47d0bec-3e0d-40da-a1dc-778929777ff9)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/e1df322b-64b3-4df4-aebd-1b988d0a3f1d)

```
print('Misclassified samples: %d'% (test_y != prediction).sum())
```
![image](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/10563271-4bb5-4982-a60e-14bdae4bf7f4)

```
data.shape
```
![image](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/b6360dd8-f20b-4725-beb6-e521104250b7)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![Screenshot 2024-04-16 110341](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/59fbf162-3832-4754-8683-2541a4bea31c)

```
tips.time.unique()
```
![Screenshot 2024-04-16 110346](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/c9abd388-d476-4186-aa2b-758b5a7e6969)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![Screenshot 2024-04-16 110351](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/b78568e8-27b3-43c8-8109-676117f546e3)

```

chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![Screenshot 2024-04-16 110356](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/96f4d769-940e-474f-8c29-bca4bab21e6c)

```

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![Screenshot 2024-04-16 110334](https://github.com/chandrumathiyazhagan/EXNO-4-DS/assets/119393023/e39d3b74-5488-4bb8-b078-a5e5d0643ff9)

# RESULT:

Thus, Feature selection and Feature scaling has been used on thegiven dataset.
