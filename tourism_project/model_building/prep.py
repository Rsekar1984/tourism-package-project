import pandas as pd, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

api   = HfApi(token=os.environ.get('HF_TOKEN'))
REPO  = 'rknv1984/tourism-dataset'

# Relative path works in both Colab and GitHub Actions
df = pd.read_csv('tourism_project/data/tourism.csv')
print(f'Loaded: {df.shape}')

df.drop(columns=['CustomerID'], inplace=True)

for col in df.columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)
print('Missing values handled.')

le = LabelEncoder()
# Alphabetical LabelEncoder mapping (must match app.py)
df['TypeofContact']  = le.fit_transform(df['TypeofContact'])   # Company Invited=0, Self Enquiry=1
df['Occupation']     = le.fit_transform(df['Occupation'])       # Free Lancer=0,Large Business=1,Self Employed=2,Small Business=3,Salaried=4
df['Gender']         = le.fit_transform(df['Gender'])           # Female=0, Male=1
df['MaritalStatus']  = le.fit_transform(df['MaritalStatus'])    # Divorced=0, Married=1, Single=2
df['Designation']    = le.fit_transform(df['Designation'])      # AVP=0, Executive=1, Manager=2, Senior Manager=3, VP=4
df['ProductPitched'] = le.fit_transform(df['ProductPitched'])   # Basic=0, Deluxe=1, King=2, Standard=3, Super Deluxe=4
print('Categorical columns encoded.')

X = df.drop(columns=['ProdTaken']).reset_index(drop=True)
y = df['ProdTaken'].reset_index(drop=True)
print(f'Features: {list(X.columns)}')

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Train: {Xtrain.shape} | Test: {Xtest.shape}')

# Save locally — index=False prevents Unnamed:0 column on reload
Xtrain.to_csv('tourism_project/data/Xtrain.csv', index=False)
Xtest.to_csv('tourism_project/data/Xtest.csv',   index=False)
ytrain.to_csv('tourism_project/data/ytrain.csv', index=False)
ytest.to_csv('tourism_project/data/ytest.csv',   index=False)
print('Splits saved to tourism_project/data/')

for f in ['Xtrain.csv', 'Xtest.csv', 'ytrain.csv', 'ytest.csv']:
    api.upload_file(
        path_or_fileobj=f'tourism_project/data/{f}',
        path_in_repo=f,
        repo_id=REPO,
        repo_type='dataset',
    )
    print(f'  Uploaded {f}')
print('All splits uploaded to HF Hub.')
