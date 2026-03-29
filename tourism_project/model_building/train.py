import pandas as pd, os, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo

api = HfApi()

print('Loading data from HF Hub...')
Xtrain = pd.read_csv('hf://datasets/rknv1984/tourism-dataset/Xtrain.csv')
Xtest  = pd.read_csv('hf://datasets/rknv1984/tourism-dataset/Xtest.csv')
ytrain = pd.read_csv('hf://datasets/rknv1984/tourism-dataset/ytrain.csv').squeeze()
ytest  = pd.read_csv('hf://datasets/rknv1984/tourism-dataset/ytest.csv').squeeze()
print(f'Train: {Xtrain.shape} | Test: {Xtest.shape}')

class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print(f'scale_pos_weight: {class_weight:.2f}')

model_pipeline = make_pipeline(
    StandardScaler(),
    xgb.XGBClassifier(
        scale_pos_weight=class_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
)

param_grid = {
    'xgbclassifier__n_estimators':     [50, 100],
    'xgbclassifier__max_depth':        [3, 4],
    'xgbclassifier__learning_rate':    [0.05, 0.1],
    'xgbclassifier__colsample_bytree': [0.5, 0.6],
    'xgbclassifier__reg_lambda':       [0.4, 0.5],
}

gs = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
gs.fit(Xtrain, ytrain)
print(f'Best params: {gs.best_params_}')

best_model = gs.best_estimator_
y_pred = (best_model.predict_proba(Xtest)[:, 1] >= 0.45).astype(int)
rep    = classification_report(ytest, y_pred, output_dict=True)
print(classification_report(ytest, y_pred))
print(f"Test Accuracy: {rep['accuracy']:.4f} | Recall: {rep['1']['recall']:.4f}")

model_path = 'best-tourism-model-v1.joblib'
joblib.dump(best_model, model_path)
print(f'Model saved: {model_path}')

REPO_ID = 'rknv1984/tourism-project-model'
try:
    api.repo_info(repo_id=REPO_ID, repo_type='model')
except RepositoryNotFoundError:
    create_repo(repo_id=REPO_ID, repo_type='model', private=False)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=REPO_ID,
    repo_type='model',
)
print('Model uploaded to HF Hub:', REPO_ID)
