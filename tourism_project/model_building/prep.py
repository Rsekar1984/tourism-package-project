import pandas as pd, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

api  = HfApi(token=os.environ.get('HF_TOKEN'))
REPO = 'rknv1984/tourism-dataset'

df = pd.read_csv('tourism_project/data/tourism.csv')
print(f'Loaded: {df.shape}')

df.drop(columns=['CustomerID'], inplace=True)
df.reset_index(drop=True, inplace=True)

for col in df.columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)
print('Missing values handled.')

le = LabelEncoder()
df['TypeofContact']  = le.fit_transform(df['TypeofContact'])
df['Occupation']     = le.fit_transform(df['Occupation'])
df['Gender']         = le.fit_transform(df['Gender'])
df['MaritalStatus']  = le.fit_transform(df['MaritalStatus'])
df['Designation']    = le.fit_transform(df['Designation'])
df['ProductPitched'] = le.fit_transform(df['ProductPitched'])
print('Categorical columns encoded.')

X = df.drop(columns=['ProdTaken'])
y = df['ProdTaken']
print(f'Features ({len(X.columns)}): {list(X.columns)}')

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f'Train: {Xtrain.shape} | Test: {Xtest.shape}')

Xtrain.to_csv('tourism_project/data/Xtrain.csv', index=False)
Xtest.to_csv('tourism_project/data/Xtest.csv',   index=False)
ytrain.to_csv('tourism_project/data/ytrain.csv', index=False)
ytest.to_csv('tourism_project/data/ytest.csv',   index=False)
print('Splits saved.')

for f in ['Xtrain.csv', 'Xtest.csv', 'ytrain.csv', 'ytest.csv']:
    api.upload_file(
        path_or_fileobj=f'tourism_project/data/{f}',
        path_in_repo=f,
        repo_id=REPO,
        repo_type='dataset',
    )
    print(f'  Uploaded {f}')
print('All splits uploaded to HF Hub.')
