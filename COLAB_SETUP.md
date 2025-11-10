# Google Colab Setup Guide

This guide explains how to use Google Colab for training and data pipelining while keeping your code in sync with this repository.

## 🚀 Quick Setup

### Option 1: Clone Repository in Colab (Recommended)

1. **Open a new Colab notebook**
2. **Mount Google Drive** (optional, for persistent storage):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Clone the repository**:
   ```python
   !git clone <your-repo-url> /content/ChessEngine
   %cd /content/ChessEngine
   ```

4. **Install dependencies**:
   ```python
   !pip install -r requirements.txt
   ```

5. **Verify setup**:
   ```python
   import chess
   import torch
   import numpy as np
   print("Setup complete!")
   ```

### Option 2: Upload Files to Colab

If you prefer not to use git in Colab:

1. **Upload the repository as a ZIP** to Colab
2. **Extract it**:
   ```python
   !unzip ChessEngine.zip
   %cd ChessEngine
   !pip install -r requirements.txt
   ```

## 📁 Working with Data and Models

### Syncing Data/Models with Google Drive

**Upload to Drive** (from Colab):
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy data to Drive
!cp -r /content/ChessEngine/data /content/drive/MyDrive/ChessEngine/
!cp -r /content/ChessEngine/models /content/drive/MyDrive/ChessEngine/
!cp -r /content/ChessEngine/logs /content/drive/MyDrive/ChessEngine/
```

**Download from Drive** (to local repo):
```python
# In Colab
!cp -r /content/drive/MyDrive/ChessEngine/data /content/ChessEngine/
!cp -r /content/drive/MyDrive/ChessEngine/models /content/ChessEngine/
!cp -r /content/drive/MyDrive/ChessEngine/logs /content/ChessEngine/
```

### Using Git to Sync Code Changes

**Pull latest changes** (in Colab):
```python
%cd /content/ChessEngine
!git pull origin main
```

**Commit and push** (if you have write access):
```python
!git add .
!git commit -m "Update from Colab"
!git push origin main
```

## 🔄 Recommended Workflow

### For Person 3 (Data Pipeline & Training):

1. **Local Development**:
   - Write code locally: `data_pipeline.py`, `train_stub.py`
   - Test with small datasets
   - Commit to git

2. **Colab Training**:
   - Clone repo in Colab
   - Run training with GPU
   - Save models/data to Google Drive

3. **Sync Back**:
   - Download models/data from Drive to local repo
   - Or use git to pull code updates

## 📝 Colab Notebook Template

See `colab_training_template.ipynb` for a ready-to-use template.

## ⚠️ Important Notes

- **Colab sessions are temporary**: Save important data to Google Drive
- **Git credentials**: You may need to set up git credentials in Colab for pushing
- **File paths**: Update paths in `config.py` if needed for Colab
- **GPU**: Enable GPU in Colab: Runtime → Change runtime type → GPU

## 🛠️ Helper Scripts

Use `sync_colab.py` to help sync files between Colab and local repo.

