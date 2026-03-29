from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

api = HfApi(token=os.environ.get('HF_TOKEN'))
repo_id = 'rknv1984/tourism-dataset'

try:
    api.repo_info(repo_id=repo_id, repo_type='dataset')
    print(f'Repo {repo_id} already exists.')
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type='dataset', private=False)
    print(f'Repo {repo_id} created.')

api.upload_file(
    path_or_fileobj='/content/tourism_project/data/tourism.csv',
    path_in_repo='tourism.csv',
    repo_id=repo_id,
    repo_type='dataset',
)
print('tourism.csv uploaded to HF Hub successfully.')
