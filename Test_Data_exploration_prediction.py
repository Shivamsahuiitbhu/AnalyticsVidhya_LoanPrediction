# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:03:53 2018

@author: Lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fd = pd.read_csv("test.csv")

fd.apply(lambda x: sum(x.isnull()),axis = 0)

fd['LoanAmount'].fillna(fd['LoanAmount'].mean(),inplace=True)

from scipy.stats import mode

loanamount_mode = mode(fd['Loan_Amount_Term']).mode[0]
fd['Loan_Amount_Term'].fillna(loanamount_mode,inplace=True)

fd['Self_Employed'].value_counts()
fd['Self_Employed'].fillna('No',inplace=True)

fd.loc[ (pd.isnull(fd['Gender'])) & (fd['Married'] =='Yes'), 'Gender'] = 'Male'
fd.loc[ (pd.isnull(fd['Gender'])) & (fd['Married'] =='No'), 'Gender'] = 'Female'

fd.loc[ (pd.isnull(fd['Dependents'])) & (fd['Married'] =='No'), 'Dependents'] = 0


Dependents = fd['Dependents']
Married = fd['Married']
Gender = fd['Gender']

y=Dependents.isnull()
for i in range(0,len(Dependents)):
    if(y[i]==True and Gender[i]=='Male'):
        Dependents[i]= '2'
    elif(y[i]==True and Gender[i]=='Female'):
        Dependents[i]= '1'

del(fd['Dependents'])
fd=fd.merge(Dependents.to_frame(),right_index=True,left_index=True)

Credit_History_mode = mode(fd['Credit_History']).mode[0]
fd['Credit_History'].fillna(Credit_History_mode,inplace=True)

fd.apply(lambda x: sum(x.isnull()),axis = 0)


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

columns = ['Loan_ID','Gender','Married','Education', 'Self_Employed','Property_Area']

for col in columns:
        fd[col]=le.fit_transform(fd[col])
        
Dependents = fd['Dependents']

for i in range(0,len(Dependents)):
    if(Dependents[i]=='0'):
        Dependents[i] = 1
    elif(Dependents[i]=='1'):
        Dependents[i] = 2
    elif(Dependents[i]=='2'):
        Dependents[i] = 3
    elif(Dependents[i]=='3+'):
        Dependents[i] = 4
    elif(Dependents[i]==0):
        Dependents[i] = 1
        
Dependents = pd.to_numeric(Dependents)
del(fd['Dependents'])
fd = fd.merge(Dependents.to_frame(),right_index=True,left_index=True)

fd['Log_LoanAmount']  = np.log(fd['LoanAmount'])

fd['TotalIncome'] = fd['ApplicantIncome'] + fd['CoapplicantIncome']
fd['TotalIncome_log'] = np.log(fd['TotalIncome'])

fd['Avg_Income']= np.divide(fd['TotalIncome'],fd['Dependents'])
fd['Avg_Income_log'] = np.log(fd['Avg_Income'])

fd['Ratio'] = np.divide(fd['TotalIncome'],fd['LoanAmount'])

fd['Ratio2'] = np.divide(fd['TotalIncome'],fd['Loan_Amount_Term'])

del(fd['LoanAmount'])
del(fd['CoapplicantIncome'])
del(fd['ApplicantIncome'])
del(fd['TotalIncome'])
del(fd['Avg_Income'])


pred =  best_random.predict(fd) 
predt = np.empty(shape=(len(pred),1),dtype=str)
c = pred>=1
for i in range(0,len(pred)):
    if(c[i]==True):
        predt[i] = 'Y'
    else:
        predt[i]= 'N'
        
sol = pd.DataFrame(predt)
pieces = [fd['Loan_ID'],sol]
sol= pd.concat(pieces,axis = 1)
sol.columns.values[1] = 'Loan_Status'
sol.to_csv("summit_fr_randomsearch.csv",header = True,encoding = 'utf-8',index=False)

