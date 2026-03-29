import pandas as pd, os, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import mlflow

# MLflow tracking server started by GitHub Actions
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('mlops-training-experiment')

# HF API token injected by GitHub Actions secret
api = HfApi()

# Load splits from HF Hub (available after prep.py runs in CI/CD)
Xtrain = pd.read_csv('hf://datasets/rknv1984/tourism-dataset/Xtrain.csv')
Xtest  = pd.read_csv('hf://datasets/rknv1984/tourism-dataset/Xtest.csv')
ytrain = pd.read_csv('hf://datasets/rknv1984/tourism-dataset/ytrain.csv').squeeze()
ytest  = pd.read_csv('hf://datasets/rknv1984/tourism-dataset/ytest.csv').squeeze()
print(f'Data loaded — Train: {Xtrain.shape}, Test: {Xtest.shape}')

class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
model_pipeline = make_pipeline(
    StandardScaler(),
    xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, eval_metric='logloss')
)
param_grid = {
    'xgbclassifier__n_estimators':     [50, 100],
    'xgbclassifier__max_depth':        [3, 4],
    'xgbclassifier__learning_rate':    [0.05, 0.1],
    'xgbclassifier__colsample_bytree': [0.5, 0.6],
    'xgbclassifier__reg_lambda':       [0.4, 0.5]
}
with mlflow.start_run():
    gs = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    gs.fit(Xtrain, ytrain)
    for i in range(len(gs.cv_results_['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(gs.cv_results_['params'][i])
            mlflow.log_metric('mean_test_score', gs.cv_results_['mean_test_score'][i])
            mlflow.log_metric('std_test_score',  gs.cv_results_['std_test_score'][i])
    mlflow.log_params(gs.best_params_)
    best_model = gs.best_estimator_
    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= 0.45).astype(int)
    y_pred_test  = (best_model.predict_proba(Xtest)[:,  1] >= 0.45).astype(int)
    train_r = classification_report(ytrain, y_pred_train, output_dict=True)
    test_r  = classification_report(ytest,  y_pred_test,  output_dict=True)
    mlflow.log_metrics({
        'train_accuracy':  train_r['accuracy'],
        'train_precision': train_r['1']['precision'],
        'train_recall':    train_r['1']['recall'],
        'train_f1_score':  train_r['1']['f1-score'],
        'test_accuracy':   test_r['accuracy'],
        'test_precision':  test_r['1']['precision'],
        'test_recall':     test_r['1']['recall'],
        'test_f1_score':   test_r['1']['f1-score'],
    })
    print(f'Test Accuracy: {test_r["accuracy"]:.4f} | Recall: {test_r["1"]["recall"]:.4f}')
    model_path = 'best-tourism-model-v1.joblib'
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path='model')
    repo_id, repo_type = 'rknv1984/tourism-project-model', 'model'
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
    except RepositoryNotFoundError:
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f'Model uploaded to HF Hub: {repo_id}')
