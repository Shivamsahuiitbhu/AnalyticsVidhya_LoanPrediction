# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the traindata
df = pd.read_csv("train.csv")

# data exploration 
df.head(5)
df.columns
df.describe()

df['ApplicantIncome'].hist(bins=50)
df.boxplot(column = 'ApplicantIncome')
# df.boxplot(column = 'ApplicantIncome',by = 'Education')


df['LoanAmount'].hist(bins=50)
df.boxplot(column = 'LoanAmount')

#  checking Loan status dependency
table1 = df.pivot_table(values = 'Loan_Status',index=['Gender','Credit_History'],aggfunc = lambda x: x.map({'Y':1,'N':0}).mean())
table6 = df.pivot_table(values = 'Loan_Status',index=['Credit_History'],aggfunc = lambda x: x.map({'Y':1,'N':0}).mean())
table1.plot(kind= 'bar',color='orange')
table6.plot(kind= 'bar',color='orange')

# check missing value in each predictor
df.apply(lambda x: sum(x.isnull()),axis=0) 

# filling LoanAmount missing value by its mean 
df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)

# library for mode func
from scipy.stats import mode

# filling Loan_Amount_term missing value by mode
loanamount_mode = mode(df['Loan_Amount_Term']).mode[0]
df['Loan_Amount_Term'].fillna(loanamount_mode,inplace=True)

# missing value credit_history = 0,1 if Loan_status = N,Y
Credit_History = df['Credit_History']
Loan_Status = df['Loan_Status']

x = Credit_History.isnull()
for i in range(0,len(Credit_History)):
    if(x[i]==True and Loan_Status[i]=='Y'):
        Credit_History[i]= 1.0
    elif(x[i]==True and Loan_Status[i]=='N'):
        Credit_History[i]= 0.0

del(df['Credit_History'])
df = df.merge(Credit_History.to_frame(),right_index=True,left_index=True)

# filling missing value by mode
df['Self_Employed'].value_counts()
df['Self_Employed'].fillna('No',inplace=True)

# filling missing value of gender and married
df.loc[ (pd.isnull(df['Married'])) & (df['Gender'] =='Male'), 'Married'] = 'Yes'
df.loc[ (pd.isnull(df['Married'])) & (df['Gender'] =='Female'), 'Married'] = 'No'

# filling missing value of Gender
df.loc[ (pd.isnull(df['Gender'])) & (df['Married'] =='Yes'), 'Gender'] = 'Male'
df.loc[ (pd.isnull(df['Gender'])) & (df['Married'] =='No'), 'Gender'] = 'Female'


# filling missing value of dependents
df.loc[ (pd.isnull(df['Dependents'])) & (df['Married'] =='No'), 'Dependents'] = 0


Dependents = df['Dependents']
Married = df['Married']
Gender = df['Gender']

y=Dependents.isnull()
for i in range(0,len(Dependents)):
    if(y[i]==True and Gender[i]=='Male'):
        Dependents[i]= '2'
    elif(x[i]==True and Gender[i]=='Female'):
        Dependents[i]= '1'

del(df['Dependents'])
df = df.merge(Dependents.to_frame(),right_index=True,left_index=True)


# treating extreme value
df['Log_LoanAmount']  = np.log(df['LoanAmount'])
df['Log_LoanAmount'].hist(bins=50)
df.boxplot(column='Log_LoanAmount')

df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20)

# categorical vaue to numeric
# using label encoder
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

columns = ['Gender','Married','Education', 'Self_Employed','Property_Area','Loan_Status']

for col in columns:
        df[col]=le.fit_transform(df[col])
        
Dependents = df['Dependents']

for i in range(0,len(Dependents)):
    if(Dependents[i]== '0'):
        Dependents[i] = 1
    elif(Dependents[i]==0):
        Dependents[i] = 1
    elif(Dependents[i]=='1'):
        Dependents[i] = 2
    elif(Dependents[i]=='2'):
        Dependents[i] = 3
    elif(Dependents[i]=='3+'):
        Dependents[i] = 4
        
Dependents = pd.to_numeric(Dependents)
del(df['Dependents'])
df = df.merge(Dependents.to_frame(),right_index=True,left_index=True)

# creating new feature
df['Avg_Income']= np.divide(df['TotalIncome'],df['Dependents'])
df['Avg_Income_log'] = np.log(df['Avg_Income'])

# Less the more the ratio more chance for loan
df['Ratio'] = np.divide(df['TotalIncome'],df['LoanAmount'])
df['Loan_ID']=le.fit_transform(df['Loan_ID'])
df['Ratio2'] = np.divide(df['TotalIncome'],df['Loan_Amount_Term'])

# All these variabe are used to make new feature
del(df['LoanAmount'])
del(df['CoapplicantIncome'])
del(df['ApplicantIncome'])
del(df['TotalIncome'])
del(df['Avg_Income'])


target_variable = df['Loan_Status']
del(df['Loan_Status'])
