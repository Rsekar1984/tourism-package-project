import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import mlflow

# MLflow tracking server (started by GitHub Actions)
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('mlops-training-experiment')

# HF API — token injected by GitHub Actions secret
api = HfApi()

# Load splits from HF Hub (available after prep.py runs in CI/CD)
Xtrain = pd.read_csv('hf://datasets/rknv1984/tourism-dataset/Xtrain.csv')
Xtest  = pd.read_csv('hf://datasets/rknv1984/tourism-dataset/Xtest.csv')
ytrain = pd.read_csv('hf://datasets/rknv1984/tourism-dataset/ytrain.csv').squeeze()
ytest  = pd.read_csv('hf://datasets/rknv1984/tourism-dataset/ytest.csv').squeeze()
print(f'Data loaded — Train: {Xtrain.shape}, Test: {Xtest.shape}')

# Class imbalance weight
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Pipeline: scale → XGBoost
model_pipeline = make_pipeline(
    StandardScaler(),
    xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, eval_metric='logloss')
)

# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators':     [50, 100],
    'xgbclassifier__max_depth':        [3, 4],
    'xgbclassifier__learning_rate':    [0.05, 0.1],
    'xgbclassifier__colsample_bytree': [0.5, 0.6],
    'xgbclassifier__reg_lambda':       [0.4, 0.5]
}

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results['params'][i])
            mlflow.log_metric('mean_test_score', results['mean_test_score'][i])
            mlflow.log_metric('std_test_score',  results['std_test_score'][i])

    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45
    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= classification_threshold).astype(int)
    y_pred_test  = (best_model.predict_proba(Xtest)[:, 1]  >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report  = classification_report(ytest,  y_pred_test,  output_dict=True)

    mlflow.log_metrics({
        'train_accuracy':  train_report['accuracy'],
        'train_precision': train_report['1']['precision'],
        'train_recall':    train_report['1']['recall'],
        'train_f1_score':  train_report['1']['f1-score'],
        'test_accuracy':   test_report['accuracy'],
        'test_precision':  test_report['1']['precision'],
        'test_recall':     test_report['1']['recall'],
        'test_f1_score':   test_report['1']['f1-score'],
    })
    print(f"Test Accuracy: {test_report['accuracy']:.4f} | Recall: {test_report['1']['recall']:.4f}")

    # Save model locally
    model_path = 'best-tourism-model-v1.joblib'
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path='model')

    # Upload model to HF Hub
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
