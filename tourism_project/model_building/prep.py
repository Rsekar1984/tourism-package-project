import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi, hf_hub_download

api = HfApi(token=os.environ.get('HF_TOKEN'))

# Download tourism.csv from HF Hub (works in both Colab AND GitHub Actions)
csv_path = hf_hub_download(
    repo_id='rknv1984/tourism-dataset',
    filename='tourism.csv',
    repo_type='dataset',
    token=os.environ.get('HF_TOKEN')
)
df = pd.read_csv(csv_path)
print(f'Dataset loaded. Shape: {df.shape}')

df.drop(columns='CustomerID', inplace=True)
print('Dropped CustomerID.')

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

X = df.drop(columns='ProdTaken')
y = df['ProdTaken']
print(f'Features: {X.shape} | Target: {y.value_counts().to_dict()}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Train: {X_train.shape} | Test: {X_test.shape}')

X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
print('Splits saved locally.')

for f in ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']:
    api.upload_file(
        path_or_fileobj=f,
        path_in_repo=f,
        repo_id='rknv1984/tourism-dataset',
        repo_type='dataset',
    )
    print(f'Uploaded {f} to HF Hub.')
print('All splits uploaded successfully.')
