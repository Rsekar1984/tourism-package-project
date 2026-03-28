from huggingface_hub import HfApi
import os

# Initialize HF API with token from environment
api = HfApi(token=os.environ.get('HF_TOKEN'))

# Upload entire deployment folder to HF Space
api.upload_folder(
    folder_path='tourism_project/deployment',
    repo_id='rknv1984/tourism-package-project',
    repo_type='space',
    path_in_repo='',
)
print('Deployment files uploaded to HF Space successfully.')
