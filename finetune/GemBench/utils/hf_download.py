import os
import time

# Connect to the official huggingface.co by default. A different endpoint must be set before
# huggingface_hub is imported, so this only fills in a default and never overrides an exported HF_ENDPOINT.
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
# Disable hf_transfer to avoid compatibility problems in some environments (set it to "1" to speed downloads up)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Missing dependency. Run: pip install huggingface_hub")
    exit(1)


def check_network(endpoint: str, timeout: int = 10) -> bool:
    """Check that the endpoint is reachable first, to avoid pointless retries."""
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.Request(endpoint, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except Exception as e:
        print(f"❌ Cannot connect to {endpoint} : {e}")
        return False


def download_with_retry(repo_id, repo_type, local_dir, max_retries=100, delay=5):
    """
    Download with automatic endpoint configuration and a retry loop.
    """
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"🌐 Current Hugging Face endpoint: {endpoint}")

    if not check_network(endpoint):
        print("⚠️ Endpoint unreachable. Check:")
        print(f"   1. whether the server can reach the internet (curl -I {endpoint})")
        print("   2. whether a proxy is needed:")
        print("        export http_proxy=http://<your-proxy>:<port>")
        print("        export https_proxy=http://<your-proxy>:<port>")
        print("   3. or switch endpoint: export HF_ENDPOINT=...")
        return

    os.makedirs(local_dir, exist_ok=True)

    attempt = 1
    while attempt <= max_retries:
        try:
            print(f"🚀 Downloading [{repo_id}] (attempt {attempt}/{max_retries}) ...")

            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=local_dir,
                max_workers=4,            # lower this on an unstable connection
                etag_timeout=30,          # give etag requests a longer timeout
                # Note: recent huggingface_hub resumes by default, so resume_download is unnecessary
            )

            print(f"🎉 Download complete. Files saved to: {os.path.abspath(local_dir)}")
            return

        except Exception as e:
            print(f"⚠️ Download interrupted: {type(e).__name__}: {e}")
            if attempt < max_retries:
                print(f"⏳ Retrying (resuming) in {delay}s ...\n")
                time.sleep(delay)
            else:
                print("❌ Maximum retries reached, download failed. Check your network.")
            attempt += 1


if __name__ == "__main__":
    # ==================== hard-coded configuration ====================
    # Target URL: https://huggingface.co/datasets/rjgpinel/GEMBench/tree/main
    REPO_ID = "rjgpinel/GEMBench"
    REPO_TYPE = "dataset"
    # Defaults to the BRIDGEVLA_DATA_ROOT environment variable; falls back to a repo-relative path when unset.
    LOCAL_DIR = os.path.join(
        os.environ.get(
            "BRIDGEVLA_DATA_ROOT",
            "data/bridgevla_data",
        ),
        "test",
    )
    MAX_RETRIES = 100

    download_with_retry(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=LOCAL_DIR,
        max_retries=MAX_RETRIES,
    )
