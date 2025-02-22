import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\KIIT\Downloads\diabetes.csv")
df.head()
#pedigree= chances based on family history from 0.08 to 2.42
#Units and way of measurement with column attributes for ease
df.columns = ['Pregnancies','Glucose(2hr)','BloodPress_Diastolic(mm hg)','SkinThickness(mm)','Insulin(muU/ml)','BMI','DiabetesPedigreeFunction','Age','Outcome']
df.duplicated().sum()
df.describe()
#unusual : skin thickness , BP , Glucose , BMI cant be zero for a living person
#INSULIN levels can drop to zero for diabetic people but not for NON-DIABETIC
sample_df = df.replace(0,None)
sample_df.info()
#mean of skin thickness & insulin will be affected by large no. of Non-recorded zero values
print("      -------------------     ")
print(sample_df['SkinThickness(mm)'].mean())
print(sample_df['Insulin(muU/ml)'].mean())
df['Glucose(2hr)'].mask(df['Glucose(2hr)']==0,120.894531,inplace=True)
df['BloodPress_Diastolic(mm hg)'].mask(df['BloodPress_Diastolic(mm hg)']==0,69.105469,inplace=True)
df['BMI'].mask(df['BMI']==0,31.992578,inplace=True)
df['SkinThickness(mm)'].mask(df['SkinThickness(mm)']==0,29.153419,inplace=True)
df['Insulin(muU/ml)'].mask((df['Insulin(muU/ml)']==0) & (df['Outcome']==0),155.548223,inplace=True)
q1_bp = np.percentile(df['BloodPress_Diastolic(mm hg)'],25)
q3_bp = np.percentile(df['BloodPress_Diastolic(mm hg)'],75)
print(q1_bp,q3_bp)
q1_i = np.percentile(df['Insulin(muU/ml)'],25)
q3_i = np.percentile(df['Insulin(muU/ml)'],75)
print(q1_i,q3_i)
q1_s = np.percentile(df['SkinThickness(mm)'],25)
q3_s = np.percentile(df['SkinThickness(mm)'],75)
print(q1_s,q3_s)
q1_d = np.percentile(df['DiabetesPedigreeFunction'],25)
q3_d = np.percentile(df['DiabetesPedigreeFunction'],75)
print(q1_d,q3_d)
#Percentiles for calculation of box plot range
outrm_df = df.loc[(df['BloodPress_Diastolic(mm hg)'] > 40) & (df['BloodPress_Diastolic(mm hg)'] < 104) &
    (df['SkinThickness(mm)'] > 14.5) & (df['SkinThickness(mm)'] < 42.6) &
    (df['Insulin(muU/ml)'] > 56.0) & (df['Insulin(muU/ml)'] < 155.548223) &
    (df['BMI'] < 50) & (df['DiabetesPedigreeFunction'] < 1.2)]
#outlier filtering


bins1 = [0,12,17,25,35,60,110]
labels1 = ['child','teen','youth','middle age(early)','middle age(late)','senior citizen']
bins2 = [0,140,300]
labels2 = ['Normal','Pre-diabetic and diabetic']
bins3 = [0,18.5,25,30,68]
labels3 = ['Underweight','healthy weight','over weight','obesity']
age_group = pd.cut(df['Age'],bins1,labels=labels1)
glucose_scales = pd.cut(df['Glucose(2hr)'],bins2,labels=labels2)
BMI_cat = pd.cut(df['BMI'],bins3,labels =labels3)
#older people tend to be more diabetic , particularly late middle age people
#glucose levels cant be below 140 range and still be referred as diabetic (this shows disparency in surveical data)
#our surveyed population is obese in majority (a noise in interference of data)
#glucose ranging and BMI ranging is observed to be not soo acurate soo not taking account of them in prediction
from sklearn.preprocessing import Normalizer
norm = Normalizer()
normalised_df = pd.DataFrame(norm.fit_transform(outrm_df))
normalised_df.columns = df.columns
sns.heatmap(normalised_df.corr(),annot=True,cmap='mako')
#Doing predictions with the models now
outrm_df = outrm_df.drop('Pregnancies',axis=1)
#RANDOMFORESTClass

from sklearn.model_selection import train_test_split
X = outrm_df[outrm_df.columns[:-1]]
y = outrm_df[outrm_df.columns[-1]]
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.20)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=500)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
from sklearn import metrics
metrics.accuracy_score(y_test,y_pred)
#KNC
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
y_pred1 = knn.predict(X_test)
metrics.accuracy_score(y_test,y_pred1)
#Logistic regression
normalised_df = normalised_df.drop('Pregnancies',axis=1)
from sklearn.model_selection import train_test_split
X2 = normalised_df[normalised_df.columns[:-1]]
y2 = outrm_df[outrm_df.columns[-1]]
X_train2,X_test2,y_train2,y_test2= train_test_split(X2,y2,test_size=0.20)
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(X_train2,y_train2)
y_pred2 = reg.predict(X_test2)
metrics.accuracy_score(y_test2,y_pred2)
import xgboost as xgb
xgb_train = xgb.DMatrix(X_train2,y_train2)
xgb_test = xgb.DMatrix(X_test2,y_test2)
params = {'objective': 'binary:logistic','max_depth': 3,'learning_rate': 0.1,}
mod = xgb.train(params=params,dtrain=xgb_train,num_boost_round=50)
pred = mod.predict(xgb_test)
pred = pred.astype("int64")
metrics.accuracy_score(y_test,pred)
params2 = {"learning_rate": 0.1,"max_depth": 3,"num_parallel_tree": 500,"objective": "binary:logistic","tree_method": "hist"}
mod2 = xgb.train(params=params2,dtrain=xgb_train,num_boost_round=50)
pred2 = mod2.predict(xgb_test)
pred2 = pred2.astype("int64")
metrics.accuracy_score(y_test,pred2)
details = input("Enter the details separated by commas: ")
details = np.array([float(x) for x in details.split(',')])
details = details.reshape(1, -1)
prediction = clf.predict(details)
prediction
import streamlit as st
st.title("🩺 Diabetes Prediction System")
st.sidebar.header("Enter Patient Data")
glucose = st.sidebar.slider("Glucose Level", 0, 200, 120)
bp = st.sidebar.slider("Blood Pressure", 0, 130, 70)
skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin Level", 0, 800, 30)
bmi = st.sidebar.slider("BMI", 0.0, 50.0, 25.0)
pedigree = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.slider("Age", 0, 100, 30)

def predict_diabetes():
    input_data = np.array([[glucose, bp, skin_thickness, insulin, bmi, pedigree, age]])
    input_scaled = norm.transform(input_data)
    prediction = clf.predict(input_scaled)[0]
    return "Diabetic" if prediction == 1 else "Not Diabetic"

# Predict button
if st.sidebar.button("Predict"):
    result = predict_diabetes()
    st.subheader(f"🩺 Prediction: **{result}**")
