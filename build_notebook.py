import json
from pathlib import Path
cells=[]
def md(s): cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(True)})
def code(s): cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':s.splitlines(True)})
md('''# Bank Marketing Subscription Prediction

**Problem:** predict whether a customer subscribed to a term deposit (`y`) from information available before a marketing call. This is binary classification.

Questions: (1) Can pre-call attributes predict subscription? (2) Which model performs best across suitable metrics, relative to a majority baseline? (3) Which features contribute most?

The notebook uses only the provided `bank-full (1).csv` (semicolon delimited) and never modifies it. The data appears to be the Bank Marketing dataset from a Portuguese bank campaign. Original source: [UCI Machine Learning Repository, Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing). Associated study: Moro, Cortez, and Rita (2014), DOI 10.1016/j.dss.2014.03.001.

We exclude `duration` from predictive features because call length is only known during/after a call; it cannot help choose whom to contact beforehand. We still examine it in EDA. This leaves 15 input features for modeling.
''')
code('''from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.inspection import permutation_importance
RANDOM_STATE=42
sns.set_theme(style='whitegrid')
''')
md('## 1. Load and audit\n`unknown` is an explicit dataset category, not a blank cell. The value `pdays = -1` means the customer was not previously contacted; it is retained as a meaningful sentinel.')
code('''DATA_PATH=Path('bank-full (1).csv')
if not DATA_PATH.exists(): raise FileNotFoundError('Place bank-full (1).csv beside this notebook.')
df=pd.read_csv(DATA_PATH, sep=';')
print('Shape:', df.shape, '| Target: y | Independent columns:', df.shape[1]-1)
display(df.head(), df.dtypes.rename('dtype').to_frame())
print('Missing values:'); display(df.isna().sum().to_frame('count'))
print('Exact duplicate rows:', df.duplicated().sum())
display(df.describe(include='all').T)
print('Target distribution:'); display(pd.DataFrame({'count':df.y.value_counts(),'share':df.y.value_counts(normalize=True)}))
''')
md('## 2. EDA\nIQR flags are a screening aid, not proof of bad data. Valid extreme values are retained. Numeric fields include `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, and `previous`; remaining fields are categorical.')
code('''num=df.select_dtypes(include='number').columns.tolist()
cat=df.select_dtypes(exclude='number').columns.tolist()
# Target balance
sns.countplot(data=df, x='y', order=['no','yes']); plt.title('Target distribution'); plt.show()
print(f"Positive class share: {(df.y=='yes').mean():.1%}; accuracy alone can be misleading.")
# Numeric distributions and IQR outlier counts
display(df[num].describe().T)
rows=[]
for c in num:
 q1,q3=df[c].quantile([.25,.75]); iqr=q3-q1
 rows.append((c,int(((df[c]<q1-1.5*iqr)|(df[c]>q3+1.5*iqr)).sum())))
display(pd.DataFrame(rows,columns=['numeric feature','IQR flagged rows']).set_index('numeric feature'))
df[num].hist(bins=30,figsize=(14,10)); plt.suptitle('Numeric feature distributions'); plt.tight_layout(); plt.show()
fig,axs=plt.subplots(5,2,figsize=(14,20))
for ax,c in zip(axs.flat,cat): sns.countplot(data=df,y=c,order=df[c].value_counts().head(15).index,ax=ax,color='steelblue'); ax.set_title(c)
plt.tight_layout(); plt.show()
fig,axs=plt.subplots(4,2,figsize=(13,13))
for ax,c in zip(axs.flat,num): sns.boxplot(x=df[c],ax=ax,showfliers=False); ax.set_title(c)
plt.tight_layout(); plt.show()
sns.heatmap(df[num].corr(),annot=True,fmt='.2f',cmap='coolwarm',center=0); plt.title('Numeric correlations'); plt.show()
fig,axs=plt.subplots(1,3,figsize=(15,4))
for ax,c in zip(axs,['job','contact','poutcome']):
 rate=df.groupby(c,observed=False).y.apply(lambda s:(s=='yes').mean()).sort_values()
 rate.plot.barh(ax=ax); ax.set_title('Subscription share by '+c); ax.set_xlabel('Share subscribed')
plt.tight_layout(); plt.show()
''')
md('**EDA interpretation:** use the charts and displayed tables to describe class imbalance, skew, correlation, and differences in subscription shares. These are associations, not causal effects. IQR outliers are retained unless they can be shown to be data errors; `pdays=-1` is a documented sentinel.')
md("## 3. Preprocessing and data split\nAt the student's request, we use a stratified 80/20 split. This differs from the teacher rubric's standard 70/30 split. Imputation, one-hot encoding, and scaling are placed inside pipelines, so they are fitted from training folds only. Exact duplicates are removed from a modeling copy, leaving the original data unchanged. No outlier removal is applied because extreme values can be real.")
code('''model_df=df.drop_duplicates().copy()
X=model_df.drop(columns=['y','duration']) # duration unavailable for pre-call decisions
y=(model_df.y=='yes').astype(int)
num_features=X.select_dtypes(include='number').columns.tolist()
cat_features=X.select_dtypes(exclude='number').columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print('Train/test:',X_train.shape,X_test.shape,'| model features:',X.shape[1])
num_pipe=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
cat_pipe=Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])
prep=ColumnTransformer([('numeric',num_pipe,num_features),('categorical',cat_pipe,cat_features)],sparse_threshold=0)
''')
md('## 4. Baseline, models, and tuning\nThe majority-class DummyClassifier is the baseline. We compare six classifiers: Logistic Regression, Decision Tree, Random Forest, SVM, KNN, and HistGradientBoosting. A small logistic-regression C grid is selected by stratified three-fold cross-validation using average precision. Test data is reserved for final evaluation.')
code('''estimators={
 'Majority baseline':DummyClassifier(strategy='most_frequent'),
 'Logistic Regression':LogisticRegression(max_iter=1500,class_weight='balanced',random_state=42),
 'Decision Tree':DecisionTreeClassifier(class_weight='balanced',min_samples_leaf=10,random_state=42),
 'Random Forest':RandomForestClassifier(n_estimators=250,min_samples_leaf=3,class_weight='balanced_subsample',n_jobs=-1,random_state=42),
 'Linear SVM':LinearSVC(C=1,class_weight='balanced',random_state=42,max_iter=3000),
 'KNN':KNeighborsClassifier(n_neighbors=15,weights='distance'),
 'HistGradientBoosting':HistGradientBoostingClassifier(max_iter=150,learning_rate=.1,random_state=42)}
cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
tune=GridSearchCV(Pipeline([('prep',prep),('model',estimators['Logistic Regression'])]),
 {'model__C':[.1,1,10]},scoring='average_precision',cv=cv,n_jobs=-1)
tune.fit(X_train,y_train)
print('Tuned logistic parameter:',tune.best_params_,'CV average precision:',round(tune.best_score_,4))
fitted={}
for name,est in estimators.items():
 if name=='Logistic Regression': fitted['Logistic Regression (tuned)']=tune.best_estimator_; continue
 p=Pipeline([('prep',prep),('model',est)]); p.fit(X_train,y_train); fitted[name]=p
''')
md('## 5. Test evaluation and comparison\nWe report accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, and PR-AUC (average precision). PR-AUC is useful with the minority positive class. Confusion matrices use class order no, yes.')
code('''def score_values(model, Xv):
 if hasattr(model,'predict_proba'): return model.predict_proba(Xv)[:,1]
 return model.decision_function(Xv)
rows=[]; preds={}; scores={}
for name,m in fitted.items():
 pred=m.predict(X_test); score=score_values(m,X_test) if name!='Majority baseline' else np.zeros(len(y_test))
 preds[name]=pred; scores[name]=score
 tn,fp,fn,tp=confusion_matrix(y_test,pred,labels=[0,1]).ravel()
 rows.append({'Model':name,'Accuracy':accuracy_score(y_test,pred),'Precision':precision_score(y_test,pred,zero_division=0),
 'Recall':recall_score(y_test,pred,zero_division=0),'Specificity':tn/(tn+fp),'F1':f1_score(y_test,pred,zero_division=0),
 'ROC-AUC':roc_auc_score(y_test,score) if len(np.unique(score))>1 else np.nan,
 'PR-AUC':average_precision_score(y_test,score)})
results=pd.DataFrame(rows).set_index('Model').sort_values('PR-AUC',ascending=False)
display(results.style.format('{:.3f}'))
''')
code('''fig,axs=plt.subplots(1,2,figsize=(14,5))
for name in fitted:
 if len(np.unique(scores[name]))>1:
  RocCurveDisplay.from_predictions(y_test,scores[name],name=name,ax=axs[0])
  PrecisionRecallDisplay.from_predictions(y_test,scores[name],name=name,ax=axs[1])
axs[0].plot([0,1],[0,1],'--',color='gray'); axs[0].set_title('ROC curves'); axs[1].set_title('Precision-recall curves'); plt.tight_layout(); plt.show()
top=[n for n in results.index if n!='Majority baseline'][:3]
fig,axs=plt.subplots(1,len(top),figsize=(5*len(top),4))
if len(top)==1: axs=[axs]
for ax,name in zip(axs,top):
 sns.heatmap(confusion_matrix(y_test,preds[name],labels=[0,1]),annot=True,fmt='d',cmap='Blues',cbar=False,ax=ax)
 ax.set_title(name); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_xticklabels(['No','Yes']); ax.set_yticklabels(['No','Yes'])
plt.tight_layout(); plt.show()
results[['Precision','Recall','F1','ROC-AUC','PR-AUC']].sort_values('PR-AUC').plot.barh(figsize=(11,7),xlim=(0,1),title='Test metrics'); plt.tight_layout(); plt.show()
''')
md('## 6. Feature importance\nPermutation importance estimates how much test PR-AUC falls when one input feature is shuffled. This is a model interpretation aid, not a causal claim.')
code('''best=next(n for n in results.index if n!='Majority baseline')
imp=permutation_importance(fitted[best],X_test,y_test,scoring='average_precision',n_repeats=5,random_state=42,n_jobs=-1)
importance=pd.Series(imp.importances_mean,index=X_test.columns).sort_values(ascending=False)
display(importance.to_frame('mean PR-AUC decrease'))
importance.head(12).sort_values().plot.barh(figsize=(9,6),title='Permutation importance: '+best); plt.xlabel('Decrease in PR-AUC'); plt.tight_layout(); plt.show()
''')
md('''## 7. Results and answers
The next cell prints answers from the measured holdout results; do not report values other than those produced by running the notebook. A higher recall identifies more subscribers but may also increase false positives. Choose a model in light of campaign capacity and the precision-recall trade-off.

**Limitations:** one random historical split may not represent future campaigns or other banks; the task is predictive, not causal. The `duration` feature is excluded for realistic pre-call use. `unknown` categories and IQR-flagged values are retained. Thresholds are not optimized for campaign costs; external/time-based validation and fresh data are future improvements.
''')
code('''winner=next(n for n in results.index if n!='Majority baseline')
print('Q1 Can pre-call information predict subscription?')
print(f"Best classifier {winner}: test PR-AUC={results.loc[winner,'PR-AUC']:.3f}, F1={results.loc[winner,'F1']:.3f}, recall={results.loc[winner,'Recall']:.3f}; positive prevalence={y_test.mean():.3f}.")
print('Q2 Best model by PR-AUC:',winner)
print('Q3 Five highest permutation-importance features:',', '.join(importance.head(5).index))
print('Compare against baseline:'); display(results.loc[['Majority baseline',winner],['Accuracy','Recall','F1','PR-AUC']])
# Preserve the split for repeatable follow-up analysis; original CSV stays unchanged.
train_export=X_train.copy(); train_export['y']=y_train.values
test_export=X_test.copy(); test_export['y']=y_test.values
Path('Dataset').mkdir(exist_ok=True)
train_export.to_csv('Dataset/train_split.csv',index=False)
test_export.to_csv('Dataset/test_split.csv',index=False)
results.to_csv('model_comparison.csv')
print('Saved train and held-out test partitions in Dataset/, plus model_comparison.csv.')
''')
md('## 8. Try a prediction yourself\nRun the code cell below after the training cells have finished. It will show a text prompt. Paste one customer as a JSON object, using all 15 input fields (numbers for numeric fields and quoted text for categories). The example is based on a held-out row; edit its values to try another case. Do not include `y` or `duration`. Run this cell again to test another customer.')
code('''import json

example_customer = json.loads(X_test.iloc[[0]].to_json(orient='records'))[0]
print('Example input (edit the values to try another customer):')
print(json.dumps(example_customer))
customer_text = input('Paste one customer JSON object, then press Enter: ')
try:
    customer = json.loads(customer_text)
    if not isinstance(customer, dict):
        raise ValueError('Enter one JSON object.')
    missing = sorted(set(X.columns) - set(customer))
    extra = sorted(set(customer) - set(X.columns))
    if missing or extra:
        raise ValueError(f'Missing fields: {missing}; unexpected fields: {extra}')
    one_customer = pd.DataFrame([customer], columns=X.columns)
    predicted = best_model.predict(one_customer)[0]
    print('Prediction:', 'SUBSCRIBE (yes)' if predicted == 1 else 'NOT SUBSCRIBE (no)')
    if hasattr(best_model, 'predict_proba'):
        probability = best_model.predict_proba(one_customer)[0, 1]
        print(f'Estimated probability of subscription: {probability:.1%}')
    print('This is a model estimate, not a guarantee.')
except Exception as err:
    print('Could not make prediction:', err)
''')
md("## Reproducibility\nRun cells from top to bottom in the same directory as `bank-full (1).csv`. Seed 42; stratified 80/20 split (student-requested deviation from the rubric's 70/30 default); stratified 3-fold logistic tuning. Dependencies are listed in `requirements.txt`.")
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}},'nbformat':4,'nbformat_minor':5}
# Interactive input is intended for a live notebook frontend; skip it during batch execution.
for cell in nb['cells']:
    if cell['cell_type']=='code' and "customer_text = input(" in ''.join(cell['source']):
        cell['metadata']['tags']=['skip-execution']
Path('ML_Project.ipynb').write_text(json.dumps(nb,indent=1),encoding='utf-8')
