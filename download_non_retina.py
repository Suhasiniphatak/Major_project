import requests, os

def download_random(save_dir, count):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(count):
        # Use Unsplash’s curated “Nature” collection to avoid 404s
        url = 'https://source.unsplash.com/collection/190727/800x600'
        try:
            # Let requests follow the redirect to the actual image
            r = requests.get(url, timeout=10, allow_redirects=True)
            if r.status_code == 200:
                with open(os.path.join(save_dir, f'{i}.jpg'), 'wb') as f:
                    f.write(r.content)
                print(f'Downloaded {i+1}/{count}')
            else:
                print(f'Error {r.status_code} at {i+1}')
        except Exception as e:
            print(f'Failed at {i+1}: {e}')

if __name__ == '__main__':
    # 200 images for training, 50 for validation
    download_random('dataset/train/non_retina', 200)
    download_random('dataset/val/non_retina', 50)
