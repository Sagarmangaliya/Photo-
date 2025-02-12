import os
import urllib.request

def download_file(url, dst):
    if not os.path.exists(dst):
        print(f"Downloading {url} to {dst}")
        urllib.request.urlretrieve(url, dst)

model_urls = {
    'iclight_sd15_fbc.safetensors': 'https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors',
    'realistic-vision-v51': 'https://huggingface.co/stablediffusionapi/realistic-vision-v51/resolve/main/',
    'RMBG-1.4': 'https://huggingface.co/briaai/RMBG-1.4/resolve/main/'
}

for filename, url in model_urls.items():
    if '/' in filename:
        model_dir = os.path.join('models', os.path.dirname(filename))
        os.makedirs(model_dir, exist_ok=True)
        download_file(url, os.path.join(model_dir, os.path.basename(filename)))
    else:
        download_file(url, os.path.join('models', filename))
