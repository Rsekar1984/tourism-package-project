from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

# Initialize HF API with token from environment
api = HfApi(token=os.environ.get('HF_TOKEN'))

# Target HF dataset repository
repo_id   = 'rknv1984/tourism-dataset'
repo_type = 'dataset'

# Create repo if it does not exist
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Repo '{repo_id}' created.")

# Upload tourism.csv directly from local Colab path
api.upload_file(
    path_or_fileobj='/content/tourism_project/data/tourism.csv',
    path_in_repo='tourism.csv',
    repo_id=repo_id,
    repo_type=repo_type,
)
print('tourism.csv uploaded to HF Hub successfully.')
