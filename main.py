import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
df=pd.read_csv('diabetes.csv')
print(df.head())
print(df.info)
print(df.describe())
print(df.isna().sum())
print(df.columns)
print(df.shape)
#Data Cleaning
df["SkinThickness"]=df["SkinThickness"].replace(0,df["SkinThickness"].mean())
print(df)
df["Insulin"]=df["Insulin"].replace(0,df["Insulin"].mean())
print(df)
#Data Visualization
plt.figure(figsize=(12,7))
plt.hist("Glucose",data=df,edgecolor="k")
plt.title("Glucose Histogram Plot")
plt.show()
plt.figure(figsize=(12,7))
plt.hist("BMI",data=df,edgecolor="k")
plt.title("BMI Histogram Plot")
plt.show()
plt.figure(figsize=(12,7))
plt.hist("Skin Thickness",data=df,edgecolor="k")
plt.title("Skin Thickness Histogram Plot")
plt.show()
plt.figure(figsize=(12,7))
plt.scatter("Pregnancies","Insulin",data=df)
plt.title("Pregnancies vs Insulin")
plt.xlabel("Pregnancies")
plt.ylabel("Insulin")
plt.show()
plt.figure(figsize=(12,7))
plt.scatter("SkinThickness","Insulin",data=df)
plt.title("Skin Thickness vs Insulin")
plt.xlabel("Skin Thickness")
plt.ylabel("Insulin")
plt.show()
plt.figure(figsize=(12,7))
plt.scatter("Glucose","BMI",data=df)
plt.title("Glucose vs BMI")
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.show()
#Outlier Removal
def remove_outlier(dataFrame):
    for column_name in dataFrame.columns:
        Q1=df[column_name].quantile(0.25)
        Q3=df[column_name].quantile(0.75)
        IQR=Q3-Q1
        lower_limit=Q1-1.5*IQR
        upper_limit=Q3+1.5*IQR
        print(f"{column_name} >> Lower Limit: {lower_limit} \nUpper Limit: {upper_limit}")
        dataFrame=dataFrame[(dataFrame[column_name]>lower_limit)|(dataFrame[column_name]<upper_limit)]
    return dataFrame
df=remove_outlier(df)
print(df)
print(df.shape)
#Data Splitting
x=df.drop(["Outcome"],axis=1)
y=df["Outcome"]
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)
print(x_train)
print(y_train)
print(x_train.shape)
print(x_test.shape)
logReg=LogisticRegression()
logReg.fit(x_train,y_train)
logReg.score(x_test,y_test)
predictions=logReg.predict(x_test)
cm=confusion_matrix(y_test,predictions)
print(cm)
sns.heatmap(cm,annot=True)
print(classification_report(y_test,predictions))