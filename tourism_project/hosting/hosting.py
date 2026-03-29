from huggingface_hub import HfApi
import os

api   = HfApi(token=os.environ.get('HF_TOKEN'))
SPACE = 'rknv1984/tourism-project'

# Delete the default template file so HF runs our app.py instead
try:
    api.delete_file(
        path_in_repo='streamlit_app.py',
        repo_id=SPACE,
        repo_type='space',
    )
    print('Deleted streamlit_app.py (default template file).')
except Exception:
    pass  # already deleted or does not exist

# Upload our deployment files
for filename in ['Dockerfile', 'app.py', 'requirements.txt']:
    api.upload_file(
        path_or_fileobj=f'tourism_project/deployment/{filename}',
        path_in_repo=filename,
        repo_id=SPACE,
        repo_type='space',
    )
    print(f'  Uploaded {filename}')

print('✅ All files uploaded to Space: rknv1984/tourism-project')
print('⏳ Space will rebuild in ~3-5 minutes.')
print('🔗 https://huggingface.co/spaces/rknv1984/tourism-project')
