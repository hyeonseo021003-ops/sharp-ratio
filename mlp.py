import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



class MlpDataset(Dataset):
    def __init__(self, p_bins: np.ndarray, r_tsy: np.ndarray, r_loan: np.ndarray):
        """
        p_bins: (N,50)
        r_tsy : (N,)
        r_loan: (N,)
        """
        self.p_bins = torch.tensor(p_bins, dtype=torch.float32)
        self.r_tsy  = torch.tensor(r_tsy, dtype=torch.float32)
        self.r_loan = torch.tensor(r_loan, dtype=torch.float32)

    def __len__(self):
        return self.p_bins.shape[0]

    def __getitem__(self, idx):
        return self.p_bins[idx], self.r_tsy[idx], self.r_loan[idx]


# -----------------------------

class mlp(nn.Module):
    """
    입력: p_bins (B,50), r_tsy (B,) -> concat (B,51)
    출력: a (B,) in (0,1)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 입력: 51개(50개 확률 + 1개 국채수익률) -> 출력: 64개 노드로 변환
            nn.Linear(51, 64), 
            nn.ReLU(),
            nn.Linear(64, 1),  # 최종적으로 1개의 '점수'를 냅니다.
            nn.Sigmoid()       # 점수를 0~1 사이의 확률(가중치)로 변환합니다. 
        )

    def forward(self, p_bins: torch.Tensor, r_tsy: torch.Tensor) -> torch.Tensor:
        # p_bins: (B,50)
        # r_tsy : (B,) or (B,1)
        if r_tsy.dim() == 1:
            r_tsy = r_tsy.unsqueeze(1)  # (B,) -> (B,1) 세로행렬로 만듬
        z = torch.cat([p_bins, r_tsy], dim=1)  # (B,51) dim=1: 가로로 붙이기
        a = self.net(z).squeeze(1)  # (B,) 신경망 통과 시키고 행벡터로 변환
        return a
    

model = mlp()






# 1. 데이터를 펴고 모델을 통과시켜 복구하는 함수
def get_model_predictions(model, p_bins, r_tsy):
    model.eval()
    with torch.no_grad():
        # (100, 3000, 51) -> (300000, 51)로 펴기
        p_bins_flat = p_bins.view(-1, 51)
        r_tsy_flat = r_tsy.view(-1, 1)
        
        # 모델 통과 (결과: 300000,)
        probs_flat = model(p_bins_flat, r_tsy_flat)
        
        # 원래 구조로 복구 (결과: 100, 3000)
        probs_3d = probs_flat.view(100, 3000)
    return probs_3d

# 2. 복구된 데이터를 받아 샤프지수를 구하는 함수
def calculate_batch_sharpe(probs_3d, actual_loan_ret, r_tsy, threshold=0.6):
    # (100, 3000, 1) -> (100, 3000)으로 모양 맞추기
    actual_ret = actual_loan_ret.squeeze(-1)
    rtsy = r_tsy.squeeze(-1)

    # 2. 0.6을 기준으로 Hard Decision (승인 1, 미승인 0)
    # probs > 0.6 은 True/False가 되므로 .float()를 통해 1.0/0.0으로 변환
    decisions = (probs_3d >= threshold).float()
    
    # 3. 개별 수익률 결정
    # 승인(1)인 사람은 actual_loan_ret을, 미승인(0)인 사람은 r_tsy를 가짐
    individual_returns = (decisions * actual_ret) + ((1 - decisions) * rtsy)
    
    # 4. 3000명의 평균 수익률과 표준편차 계산
    mean_ret = torch.mean(individual_returns)
    std_ret = torch.std(individual_returns)
    mean_rtsty = rtsy.mean()
    # 5. Sharpe Ratio 계산 => 평균 수익률 - 평균 국채 수익률 평균(수익률-국채수익률)
    final_sharpe = (mean_ret - mean_rtsty)/ (std_ret + 1e-8)
    
    return final_sharpe.item(), decisions



print('working!!!')

