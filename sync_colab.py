#!/usr/bin/env python3
"""
Helper script to sync files between local repository and Google Colab.
This script helps download/upload models, data, and logs.

Usage:
    python sync_colab.py download  # Download from Google Drive
    python sync_colab.py upload    # Upload to Google Drive
"""

import os
import sys
import argparse
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).parent
DRIVE_BASE = Path("/content/drive/MyDrive/ChessEngine")  # Colab path
LOCAL_BASE = REPO_ROOT  # Local path

DIRS_TO_SYNC = ["models", "data", "logs"]


def check_colab():
    """Check if running in Colab environment."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def mount_drive():
    """Mount Google Drive in Colab."""
    if not check_colab():
        print("Not running in Colab. Skipping drive mount.")
        return False
    
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        return True
    except Exception as e:
        print(f"Error mounting drive: {e}")
        return False


def download_from_drive():
    """Download files from Google Drive to local repo."""
    if not check_colab():
        print("Error: This script must be run in Colab to download from Drive.")
        print("For local downloads, manually copy files from Drive.")
        return False
    
    if not DRIVE_BASE.exists():
        print(f"Drive path not found: {DRIVE_BASE}")
        print("Mounting drive...")
        if not mount_drive():
            return False
    
    print("Downloading from Google Drive...")
    for dir_name in DIRS_TO_SYNC:
        drive_dir = DRIVE_BASE / dir_name
        local_dir = LOCAL_BASE / dir_name
        
        if drive_dir.exists():
            print(f"  Syncing {dir_name}...")
            os.makedirs(local_dir, exist_ok=True)
            os.system(f"cp -r {drive_dir}/* {local_dir}/ 2>/dev/null || true")
            print(f"  ✓ {dir_name} synced")
        else:
            print(f"  ⚠ {dir_name} not found in Drive")
    
    print("Download complete!")
    return True


def upload_to_drive():
    """Upload files from local repo to Google Drive."""
    if not check_colab():
        print("Error: This script must be run in Colab to upload to Drive.")
        print("For local uploads, manually copy files to Drive.")
        return False
    
    if not DRIVE_BASE.exists():
        print(f"Drive path not found: {DRIVE_BASE}")
        print("Mounting drive...")
        if not mount_drive():
            return False
    
    print("Uploading to Google Drive...")
    os.makedirs(DRIVE_BASE, exist_ok=True)
    
    for dir_name in DIRS_TO_SYNC:
        local_dir = LOCAL_BASE / dir_name
        drive_dir = DRIVE_BASE / dir_name
        
        if local_dir.exists() and any(local_dir.iterdir()):
            print(f"  Syncing {dir_name}...")
            os.makedirs(drive_dir, exist_ok=True)
            os.system(f"cp -r {local_dir}/* {drive_dir}/ 2>/dev/null || true")
            print(f"  ✓ {dir_name} synced")
        else:
            print(f"  ⚠ {dir_name} is empty or doesn't exist")
    
    print("Upload complete!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Sync files between local repo and Google Colab/Drive"
    )
    parser.add_argument(
        "action",
        choices=["download", "upload"],
        help="Action to perform: download from Drive or upload to Drive"
    )
    
    args = parser.parse_args()
    
    if args.action == "download":
        download_from_drive()
    elif args.action == "upload":
        upload_to_drive()


if __name__ == "__main__":
    main()

