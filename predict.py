import sys, joblib, pandas as pd
if len(sys.argv)!=2:
    raise SystemExit('Usage: python predict.py module_metrics.csv')
a=joblib.load('bug_risk_model.pkl'); df=pd.read_csv(sys.argv[1]); X=df.reindex(columns=a['features'])
prob=a['model'].predict_proba(X)[:,1]
out=df.copy(); out['bug_risk_probability']=prob; out['risk_level']=pd.cut(prob,[-1,.30,.60,1],labels=['LOW','MEDIUM','HIGH'])
out.to_csv('predictions.csv',index=False); print(out[['bug_risk_probability','risk_level']].to_string(index=False)); print('\nSaved predictions.csv')
