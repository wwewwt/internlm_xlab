import os
import torch
from modelscope import snapshot_download
os.system('pwd')
model_dir = snapshot_download('wwewwt/zuchongzhi', cache_dir='./model', revision='v1.0.0')
os.system('streamlit run app.py --server.address=0.0.0.0 --server.port 7860')
