from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo, hf_hub_download
import os

api = HfApi(token=os.environ.get('HF_TOKEN'))
repo_id = 'rknv1984/tourism-dataset'
repo_type = 'dataset'

# Create repo if not exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f'Repo {repo_id} already exists.')
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f'Repo {repo_id} created.')

# Download tourism.csv from HF Hub (already uploaded there)
csv_path = hf_hub_download(
    repo_id=repo_id,
    filename='tourism.csv',
    repo_type=repo_type,
    token=os.environ.get('HF_TOKEN')
)
print(f'tourism.csv downloaded from HF Hub: {csv_path}')

# Re-upload to confirm it is there (idempotent)
api.upload_file(
    path_or_fileobj=csv_path,
    path_in_repo='tourism.csv',
    repo_id=repo_id,
    repo_type=repo_type,
)
print('tourism.csv verified on HF Hub successfully.')
