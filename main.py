import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
import pandas as pd
from torchvision import transforms
import glob
from tqdm import tqdm
from urllib.request import urlopen
from PIL import Image

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import time

# 분산학습 library
import torch.distributed as dist

#random seed 고정
import random

seed = 40
deterministic = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
warnings.filterwarnings('ignore')

# custome py import
import Data_Load
import Model_Compression
import Data_Load
import Tokenizer
import Train_Setting

# 데이터 경로 지정
train_save_path = ...
test_save_path = ...

if __name__ == "__main__":
    # SageMaker 환경 변수
    world_size = int(os.environ.get('WORLD_SIZE', 1))  # 총 프로세스 수
    rank = int(os.environ.get('RANK', 0))  # 현재 프로세스의 순번
    local_rank = int(os.environ.get('LOCAL_RANK', 0))  # 로컬 GPU 번호

    # PyTorch 분산 초기화
    dist.init_process_group(
        backend='nccl',  # GPU 사용 시 nccl
        init_method='env://',  # SageMaker 환경 변수를 통해 초기화
        world_size=world_size,
        rank=rank
    )

    Data_Load.data_load()
    Model_Compression.peft_config()
    Model_Compression.model_build()

    # 데이터 병렬화
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    Data_Load.data_structure()
    Tokenizer.replace_padding_with_ignore(labels, padding_value=..., ignore_value=-100)
    Tokenizer.tokenizing()
    Train_Setting.training_config()
    trainer.train()



