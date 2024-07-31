import os
import requests
import sys

BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
sam2_hiera_t_url = f"{BASE_URL}sam2_hiera_tiny.pt"
sam2_hiera_s_url = f"{BASE_URL}sam2_hiera_small.pt"
sam2_hiera_b_plus_url = f"{BASE_URL}sam2_hiera_base_plus.pt"
sam2_hiera_l_url = f"{BASE_URL}sam2_hiera_large.pt"
groundingdino_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

def download_file(url, filename):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Successfully downloaded: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download from {url}: {e}")
        sys.exit(1)

def main():
    os.makedirs('checkpoints', exist_ok=True)
    os.chdir('checkpoints')

    download_file(sam2_hiera_l_url, os.path.basename(sam2_hiera_l_url))
    download_file(groundingdino_url, os.path.basename(groundingdino_url))

    os.chdir('..')

if __name__ == "__main__":
    main()