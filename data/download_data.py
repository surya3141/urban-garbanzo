"""
data/download_data.py
---------------------
Downloads the Kaggle Fake-News dataset and places it in this directory.

Option A – Kaggle API (recommended)
------------------------------------
  pip install kaggle
  # Place your kaggle.json API token in ~/.kaggle/kaggle.json
  python data/download_data.py --source kaggle

Option B – Direct CSV download (no Kaggle account needed)
----------------------------------------------------------
  python data/download_data.py --source direct

The script downloads and saves:
  data/fake.csv  – labelled fake articles
  data/true.csv  – labelled real articles
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent

# Mirror links for the CSV files (GitHub raw / Hugging Face datasets)
_DIRECT_URLS = {
    "fake.csv": (
        "https://raw.githubusercontent.com/joolsa/fake_real_news_dataset/"
        "master/fake.csv"
    ),
    "true.csv": (
        "https://raw.githubusercontent.com/joolsa/fake_real_news_dataset/"
        "master/true.csv"
    ),
}


def download_via_kaggle() -> None:
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("Install the Kaggle package first:  pip install kaggle")
        sys.exit(1)

    print("Downloading dataset from Kaggle …")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.system(
        f"kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset "
        f"-p {DATA_DIR} --unzip"
    )
    print(f"Dataset saved to {DATA_DIR}/")


def download_direct() -> None:
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print("Install requests and tqdm:  pip install requests tqdm")
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)

    for filename, url in _DIRECT_URLS.items():
        dest = DATA_DIR / filename
        if dest.exists():
            print(f"  {filename} already exists – skipping.")
            continue

        print(f"  Downloading {filename} …")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

        print(f"  Saved → {dest}")


def verify() -> None:
    ok = True
    for name in ("fake.csv", "true.csv"):
        path = DATA_DIR / name
        if path.exists():
            import pandas as pd
            df = pd.read_csv(path, nrows=1)
            print(f"  ✔ {name} – columns: {list(df.columns)}")
        else:
            print(f"  ✗ {name} – NOT FOUND")
            ok = False

    if not ok:
        print("\nSome files are missing. Check the download and try again.")
        sys.exit(1)
    else:
        print("\nAll dataset files present. Ready to train!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Fake-News dataset")
    parser.add_argument(
        "--source",
        choices=["kaggle", "direct"],
        default="direct",
        help="Where to download from (default: direct mirror)",
    )
    args = parser.parse_args()

    if args.source == "kaggle":
        download_via_kaggle()
    else:
        download_direct()

    verify()


if __name__ == "__main__":
    main()
