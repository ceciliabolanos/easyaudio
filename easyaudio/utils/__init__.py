import requests
from tqdm import tqdm
from loguru import logger

def download_blob(url, filename):
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 1024
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

        with open(filename, 'wb') as f:
            for data in response.iter_content(chunk_size=chunk_size):
                progress_bar.update(len(data))
                f.write(data)

        progress_bar.close()
        logger.info("File downloaded successfully as", filename)
    else:
        raise Exception("Failed to download file from {}".format(url))