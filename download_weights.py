import os
import html
import glob
import uuid
import hashlib
import requests
from tqdm import tqdm
from utils import files_utils
import zipfile

# code adapted from https://github.com/royorel/StyleSDF/blob/main/download_models.py


def download_file(session, info, chunk_size=128, num_attempts=10):
    file_path = f'{info["save_path"]}/{info["zip_file_name"]}'
    file_url = info['url']
    files_utils.init_folders(file_path)
    tmp_path = file_path + '.tmp.' + uuid.uuid4().hex
    progress_bar = tqdm(total=info['size'], unit='B', unit_scale=True)
    for attempts_left in reversed(range(num_attempts)):
        data_size = 0
        progress_bar.reset()
        try:
            # Download.
            data_md5 = hashlib.md5()
            with session.get(file_url, stream=True) as res:
                res.raise_for_status()
                with open(tmp_path, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=chunk_size<<10):
                        progress_bar.update(len(chunk))
                        f.write(chunk)
                        data_size += len(chunk)
                        data_md5.update(chunk)

            # Validate.
            if 'size' in info and data_size != info['size']:
                raise IOError('Incorrect file size', file_path)
            break
        except:
            # Last attempt => raise error.
            if not attempts_left:
                raise

            # Handle Google Drive virus checker nag.
            if data_size > 0 and data_size < 8192:
                with open(tmp_path, 'rb') as f:
                    data = f.read()
                links = [html.unescape(link) for link in data.decode('utf-8').split('"') if 'confirm=t' in link]
                if len(links) == 1:
                    file_url = requests.compat.urljoin(file_url, links[0])
                    continue

    progress_bar.close()

    # Rename temp file to the correct name.
    os.replace(tmp_path, file_path) # atomic

    # Attempt to clean up any leftover temps.
    for filename in glob.glob(file_path + '.tmp.*'):
        try:
            os.remove(filename)
        except:
            pass


def download_pretrained_models():
    spaghetti_chairs_large = {'url': 'https://drive.google.com/uc?id=14yWBYYdW8VjM8aIypXATLVdQaf2xQnB-',
                              'size': 111447340,
                              'zip_file_name': 'spaghetti_chairs_large.zip',
                              'save_path': './assets/checkpoints',
                              'model_name': 'spaghetti_chairs_large'}

    spaghetti_airplanes = {'url': 'https://drive.google.com/uc?id=1wmhTC9N0UMX5rqEuyzkp9KBeQf4BCDC-',
                           'size': 101827686,
                           'zip_file_name': 'spaghetti_airplanes.zip',
                           'save_path': './assets/checkpoints',
                           'model_name': 'spaghetti_airplanes'}

    info = spaghetti_airplanes
    print(f'Downloading {info["model_name"]}')
    zip_file_path = f'{info["save_path"]}/{info["zip_file_name"]}'
    with requests.Session() as session:
        download_file(session, info)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(info["save_path"])
    files_utils.delete_single(zip_file_path)
    print('Done!')



if __name__ == "__main__":
    download_pretrained_models()

    'docker run -t -d -v /disk/amirh/clipasso_docker:/home/vinker/dev --name portclipasso -p 8888:8888 --gpus all yaelvinker/clipasso_docker'