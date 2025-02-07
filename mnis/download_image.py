# MNIST를 다운받을 경로
import os
from urllib.request import urlretrieve

url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'

# MNIST를 저장할 디렉토리 (`./data/`)
dataset_dir = os.path.join(os.getcwd(), 'data')

# Pickle로 저장할 경로
save_file = dataset_dir + "/mnist.pkl"

# MNIST 데이터셋의 파일명 (딕셔너리)
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

# 해당 경로가 없을 시 디렉토리 새로 생성
os.makedirs(dataset_dir, exist_ok=True)

# 해당 경로에 존재하지 않는 파일을 모두 다운로드
for filename in key_file.values():
    if filename not in os.listdir(dataset_dir):
        urlretrieve(url + filename, os.path.join(dataset_dir, filename))
        print("Downloaded %s to %s" % (filename, dataset_dir))