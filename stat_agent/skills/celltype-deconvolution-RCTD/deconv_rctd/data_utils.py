"""Auto-download RCTD external data (likelihood tables) on first use."""

import os
import logging
from pathlib import Path
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)

# Files needed by RCTD from SpatialScope extdata
RCTD_FILES = [
    "Q_mat_1_1.txt.gz",
    "Q_mat_1_2.txt.gz",
    "Q_mat_2_1.txt.gz",
    "Q_mat_2_2.txt.gz",
    "X_vals.txt",
]

# GitHub raw URL (works without API, no rate limit issues for small number of files)
_BASE_URL = "https://raw.githubusercontent.com/YangLabHKUST/SpatialScope/master/extdata"


def get_extdata_dir():
    """Get the extdata directory path, relative to RCTD.py."""
    return Path(__file__).parent / "extdata"


def ensure_extdata():
    """Download RCTD likelihood tables if not already present.

    Files are downloaded from the SpatialScope GitHub repository
    to {package_dir}/deconv_rctd/extdata/. This only happens once
    (~350MB total).

    Returns:
        Path to the extdata directory.
    """
    ext_dir = get_extdata_dir()

    # Check if all files exist
    missing = [f for f in RCTD_FILES if not (ext_dir / f).exists()]
    if not missing:
        return str(ext_dir)

    # Create directory
    ext_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading RCTD reference data ({len(missing)} files) to {ext_dir}")
    print(f"Downloading RCTD reference data ({len(missing)} files)...")
    print("This only happens once (~350MB total).")

    for i, filename in enumerate(missing):
        url = f"{_BASE_URL}/{filename}"
        target = ext_dir / filename
        print(f"  [{i+1}/{len(missing)}] {filename}...", end=" ", flush=True)
        try:
            urllib_request.urlretrieve(url, str(target))
            size_mb = target.stat().st_size / (1024 * 1024)
            print(f"done ({size_mb:.1f} MB)")
        except (HTTPError, URLError) as e:
            print(f"FAILED: {e}")
            logger.error(f"Failed to download {url}: {e}")
            # Clean up partial file
            if target.exists():
                target.unlink()
            raise RuntimeError(
                f"Failed to download RCTD reference data: {filename}\n"
                f"URL: {url}\n"
                f"Error: {e}\n\n"
                f"You can manually download from:\n"
                f"  https://github.com/YangLabHKUST/SpatialScope/tree/master/extdata\n"
                f"and place files in: {ext_dir}"
            ) from e

    print("RCTD reference data ready.")
    logger.info("RCTD reference data download complete")
    return str(ext_dir)
