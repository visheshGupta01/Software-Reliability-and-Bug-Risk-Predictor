import os, urllib.request, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

URL='https://raw.githubusercontent.com/ApoorvaKrisna/NASA-promise-dataset-repository/main/jm1.csv'
DATA='jm1.csv'; MODEL='bug_risk_model.pkl'
if not os.path.exists(DATA):
    print('Downloading NASA/PROMISE JM1 dataset...')
    urllib.request.urlretrieve(URL, DATA)

df=pd.read_csv(DATA)
df=df.drop_duplicates().copy()
y=df['defects'].astype(str).str.strip().str.lower().map({'true':1,'false':0})
X=df.drop(columns=['defects']).apply(pd.to_numeric, errors='coerce')
valid=y.notna(); X,y=X.loc[valid],y.loc[valid]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=Pipeline([('imputer',SimpleImputer(strategy='median')),('classifier',RandomForestClassifier(n_estimators=500,class_weight='balanced',random_state=42,n_jobs=-1,min_samples_leaf=2))])
model.fit(X_train,y_train)
p=model.predict(X_test); prob=model.predict_proba(X_test)[:,1]
metrics={'accuracy':accuracy_score(y_test,p),'precision':precision_score(y_test,p,zero_division=0),'recall':recall_score(y_test,p,zero_division=0),'f1':f1_score(y_test,p,zero_division=0),'roc_auc':roc_auc_score(y_test,prob)}
artifact={'model':model,'features':list(X.columns),'metrics':metrics,'dataset':'NASA/PROMISE JM1'}
joblib.dump(artifact,MODEL)
print('\nDataset shape:',df.shape)
print('Defect distribution:',y.value_counts().to_dict())
print('\nMetrics:'); [print(f'{k}: {v:.4f}') for k,v in metrics.items()]
print('\nConfusion matrix:\n',confusion_matrix(y_test,p))
print('\n',classification_report(y_test,p,digits=4))
print('Saved:',MODEL)
