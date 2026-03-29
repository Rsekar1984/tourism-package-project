from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

api    = HfApi(token=os.environ.get('HF_TOKEN'))
REPO   = 'rknv1984/tourism-dataset'
RTYPE  = 'dataset'

try:
    api.repo_info(repo_id=REPO, repo_type=RTYPE)
    print(f'Repo {REPO} already exists.')
except RepositoryNotFoundError:
    create_repo(repo_id=REPO, repo_type=RTYPE, private=False)
    print(f'Repo {REPO} created.')

# Relative path — works in Colab AND GitHub Actions
api.upload_file(
    path_or_fileobj='tourism_project/data/tourism.csv',
    path_in_repo='tourism.csv',
    repo_id=REPO,
    repo_type=RTYPE,
)
print('✅ tourism.csv uploaded to HF Hub.')
