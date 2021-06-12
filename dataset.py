import torch
import torch.utils.data.Dataset as Dataset


class CarlaDataset(Dataset):
    def __init__(self, path):
        super(CarlaDataset, self).__init__()
        # 데이터 전처리 파트
        pass
    
    def __len__(self):
        # 데이터 셋 길이
        pass
    
    def __getitem__(self):
        # 특정 1개 샘플을 추출하는 함수
        pass