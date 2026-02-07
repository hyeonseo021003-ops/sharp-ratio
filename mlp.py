import os
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
            r_tsy = r_tsy.unsqueeze(1)  # (B,) -> (B,1)
        z = torch.cat([p_bins, r_tsy], dim=1)  # (B,51)
        a = torch.sigmoid(self.net(z)).squeeze(1)  # (B,)
        return a
    




def calculate_final_sharpe(model, p_bins, r_tsy, actual_loan_ret, threshold=0.6):
    """
    model: 학습된 mlp
    p_bins: 50개 구간 확률 [3000, 50]
    r_tsy: 국채 수익률 [3000, 1]
    actual_loan_ret: 실제 대출 수익률 [3000, 1]
    threshold: 승인 기준 (여기서는 0.6)
    """
    
    # 1. 모델로부터 0~1 사이의 연속적인 값(확률)을 받음
    model.eval() # 평가 모드 (드롭아웃 등 비활성화)
    with torch.no_grad(): # 평가 시에는 기울기 계산이 필요 없음
        probs = model(p_bins, r_tsy) # 결과: (3000,)
    
    # 2. 0.6을 기준으로 Hard Decision (승인 1, 미승인 0)
    # probs > 0.6 은 True/False가 되므로 .float()를 통해 1.0/0.0으로 변환
    decisions = (probs >= threshold).float()
    
    # 3. 개별 수익률 결정
    # 승인(1)인 사람은 actual_loan_ret을, 미승인(0)인 사람은 r_tsy를 가짐
    individual_returns = (decisions * actual_loan_ret.squeeze()) + ((1 - decisions) * r_tsy.squeeze())
    
    # 4. 3000명의 평균 수익률과 표준편차 계산
    mean_ret = torch.mean(individual_returns)
    std_ret = torch.std(individual_returns)
    
    # 5. Sharpe Ratio 계산 => 평균 수익률 - 평균 국채 수익률 평균(수익률-국채수익률)
    final_sharpe = (mean_ret - r_tsy.squeeze())/ (std_ret + 1e-8)
    
    return final_sharpe.item(), decisions

# --- 실행 예시 ---
# sharpe_val, final_decisions = calculate_final_sharpe(model, p_bins_tensor, rf_tensor, loan_ret_tensor)
# print(f"기준값 0.6 적용 시 최종 Sharpe Ratio: {sharpe_val:.4f}")
# print(f"총 승인 건수: {final_decisions.sum().item()} / 3000")
