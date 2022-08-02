import requests
from PIL import Image
import numpy as np
import io

url = 'http://127.0.0.1:8080/predict'

# with open("hairs_dataset/bun/0a9337fb54.jpg", 'rb') as f:
files = {'file': open('hairs_dataset/dreadlocks/71a6e8b468.jpg','rb')}
r = requests.post(url, files=files)

out = r.content
print(out)