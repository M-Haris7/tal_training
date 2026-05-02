### Step 1: Run Bootstrap Script
In your pod terminal, run the following command to download and execute the setup script:

```bash
curl -sL "https://gist.githubusercontent.com/M-Haris7/0a0755608df5e582cc917e5412c8f3ef/raw/a9989cb60d0077c4aef6dbda1ea37f715e1eb30b/bootstrap.sh" -o /workspace/bootstrap.sh && bash /workspace/bootstrap.sh
```

### Step 2: Activate Virtual Environment
Activate your newly created virtual environment and open your notebook:

```bash
source /workspace/tal_training/venv/bin/activate
```

### Step 3: Select the Kernel in VS Code
In VS Code, open your notebook and select the correct kernel so it uses the environment you just created:

1. Click the kernel selector in the **top-right** corner.
2. Click **"Select Another Kernel"**.
3. Choose **"Python Environments"**.
4. Select the path: `/workspace/tal_training/venv/bin/python`.


### Step 4: Save Checkpoints to Hugging Face *(Recommended)*
Before stopping the pod, push your trained checkpoints to Hugging Face so your next pod can grab them. Run this snippet directly in your terminal:

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi(token='hf_YOUR_TOKEN_HERE')  # <------- Replace the token with yours
api.upload_folder(
    folder_path='/workspace/checkpoints',
    repo_id='Sehrish05/THUMOS14',
    repo_type='dataset',
    path_in_repo='checkpoints',
)
print('Uploaded.')
"
```

### Step 5: Restore Checkpoints Automatically
To grab your checkpoints back automatically in future sessions, add this snippet to your `bootstrap.sh` file (between steps 6 and 7):

```bash
# ── 6b. Restore checkpoints from HF if missing locally ───────────────────
if [ ! -f /workspace/checkpoints/mamba_scanner_best.pth ]; then
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$HF_DATASET_REPO', repo_type='dataset',
                  allow_patterns='checkpoints/*',
                  local_dir='/workspace', token='$HF_TOKEN')
" || echo "[bootstrap] no checkpoints to restore yet"
fi
```