import pandas as pd, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

api = HfApi(token=os.environ.get('HF_TOKEN'))

# Read from LOCAL path — avoids hf:// errors
df = pd.read_csv('/content/tourism_project/data/tourism.csv')
print(f'Dataset loaded. Shape: {df.shape}')

df.drop(columns=['CustomerID'], inplace=True)
print('Dropped CustomerID.')

for col in df.columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)
print('Missing values handled.')

le = LabelEncoder()
# TypeofContact: Company Invited=0, Self Enquiry=1
df['TypeofContact'] = le.fit_transform(df['TypeofContact'])
# Occupation: alphabetical encoding
df['Occupation'] = le.fit_transform(df['Occupation'])
# Gender: Female=0, Male=1
df['Gender'] = le.fit_transform(df['Gender'])
# MaritalStatus: Divorced=0, Married=1, Single=2
df['MaritalStatus'] = le.fit_transform(df['MaritalStatus'])
# Designation: AVP=0, Executive=1, Manager=2, Senior Manager=3, VP=4
df['Designation'] = le.fit_transform(df['Designation'])
# ProductPitched: Basic=0, Deluxe=1, King=2, Standard=3, Super Deluxe=4
df['ProductPitched'] = le.fit_transform(df['ProductPitched'])
print('Categorical columns encoded.')

# reset_index(drop=True) PREVENTS Unnamed:0 column error on predict
X = df.drop(columns=['ProdTaken']).reset_index(drop=True)
y = df['ProdTaken'].reset_index(drop=True)
print(f'Features: {X.shape} | Target distribution: {y.value_counts().to_dict()}')

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Train: {Xtrain.shape} | Test: {Xtest.shape}')

Xtrain.to_csv('/content/Xtrain.csv', index=False)
Xtest.to_csv('/content/Xtest.csv',   index=False)
ytrain.to_csv('/content/ytrain.csv', index=False)
ytest.to_csv('/content/ytest.csv',   index=False)
print('Splits saved locally.')

for f in ['Xtrain.csv', 'Xtest.csv', 'ytrain.csv', 'ytest.csv']:
    api.upload_file(
        path_or_fileobj=f'/content/{f}',
        path_in_repo=f,
        repo_id='rknv1984/tourism-dataset',
        repo_type='dataset',
    )
    print(f'Uploaded {f} to HF Hub.')
print('All splits uploaded successfully.')
