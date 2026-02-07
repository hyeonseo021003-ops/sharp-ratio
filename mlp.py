import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 최적화 및 학습 (GPU 최적화 버전)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = nn.Sequential(
    nn.Linear(51, 64),
    nn.ReLU(), # 활성화 함수: 비선형성을 추가하여 복잡한 패턴 학습
    nn.Linear(64, 1), # 출력층: 64개 정보를 모아 최종적으로 1개의 '점수' 도출
    nn.Sigmoid() # 시그모이드: 점수를 0~1 사이의 '확률' 값으로 변환
).to(device)

# 최적화 도구: Adam을 사용하며, 학습률(lr)은 0.001로 설정
optimizer = optim.Adam(model.parameters(), lr=0.001)

# data_x_3d: (100, 3000, 51) -> 수익률 확률 분포와 국채 수익률
# data_y_3d: (100, 3000, 1)  -> 실제 수익률

def train_step(data_x_3d, data_y_3d, num_epochs=50):
    for epoch in range(num_epochs):
        model.train()
        
        # 1. Forward
        x_flat = data_x_3d.view(-1, 51) # (100, 3000, 51) -> (300,000, 51)
        probs_flat = model(x_flat).squeeze(-1) # 모델 통과 후 승인확률 get 1차원 형태로
        probs_3d = probs_flat.view(100, 3000) # 다시 (100팀, 3000명)으로 복구
        
        # 2. 데이터 분리 
        r_tsy_3d = data_x_3d[:, :, -1] # 국채수익률만 분리
        actual_ret_3d = data_y_3d.squeeze(-1) # (100, 3000) 차원 일치
        
        # 3. probs_3d가 1에 가까워지면 actual_ret_3d가 되고, 0에 가까워지면 r_tsy_3d가 됩니다.
        individual_returns = (probs_3d * actual_ret_3d) + ((1 - probs_3d) * r_tsy_3d)
        
        mean_ret = individual_returns.mean(dim=1) # dim=1 : 가로 방향. 한 팀 안에 있는 3,000명의 평균을 계산
        std_ret = individual_returns.std(dim=1)
        mean_rtsy = r_tsy_3d.mean(dim=1)
        
        # 배치별 샤프 지수
        batch_sharpe = (mean_ret - mean_rtsy) / (std_ret + 1e-8)
        
        # 4. Loss: 마이너스 샤프 지수의 평균. 손실함수 최적화
        loss = -batch_sharpe.mean()
        
        # 5. Optimization
        optimizer.zero_grad() # 이전 단계 기울기 초기화
        loss.backward() # 역전파 : 어떤 가중치를 고쳐야 sharp ratio가 올라가는지
        optimizer.step() # 가중치 수정 실행
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f}")

print('Ready for Training!!!')



# 1. 모든 데이터가 합쳐진 하나의 텐서를 준비 (예: total_data_3d)
# total_data_3d 모양: (100, 3000, 52)
# [0~49: 확률 bins, 50: 국채수익률, 51: 실제 대출 수익률]

total_tensor = torch.tensor(total_data_3d, dtype=torch.float32).to(device)

# 2. 입구(X) 데이터 분리: 0번부터 50번 컬럼까지 (총 51개)
data_x_3d = total_tensor[:, :, :51] 

# 3. 정답(Y) 데이터 분리: 마지막 51번 컬럼만 (총 1개)
data_y_3d = total_tensor[:, :, 51:] 

# 학습 
train_step(data_x_3d, data_y_3d, num_epochs=200)











# 1. 데이터를 펴고 모델을 통과시켜 복구하는 함수
def get_model_predictions(test_x_3d, test_y_3d, threshold=0.6):
    model.eval() # 평가모드
    with torch.no_grad(): # 기울기 계산 금지 - 메모리, 속도
        # (100, 3000, 51) -> (300000, 51)로 펴기
        x_flat = test_x_3d.view(-1, 51)
        
        # 모델 통과 (결과: 300000,)
        probs_flat = model(x_flat)
        
        # 원래 구조로 복구 (결과: 100, 3000)
        probs_3d = probs_flat.view(100, 3000)

        r_tsy_3d = test_x_3d[:, :, -1] # 국채 수익률 분리
        actual_ret_3d = test_y_3d.squeeze(-1) # 실제 수익률
    

    # 2. 0.6을 기준으로 Hard Decision (승인 1, 미승인 0)
    # probs > 0.6 은 True/False가 되므로 .float()를 통해 1.0/0.0으로 변환
    decisions = (probs_3d >= threshold).float()
    
    # 3. 개별 수익률 결정
    # 승인(1)인 사람은 actual_loan_ret을, 미승인(0)인 사람은 r_tsy를 가짐
    # 개개인의 수익률 (100, 3000)행렬을 결과값으로
    individual_returns = (decisions * actual_ret_3d) + ((1 - decisions) * r_tsy_3d)
    
    # 4. 3000명의 평균 수익률과 표준편차 계산
    mean_ret = individual_returns.mean(dim=1)   
    std_ret = individual_returns.std(dim=1)     
    mean_rtsy = r_tsy_3d.mean(dim=1)
    # 5. Sharpe Ratio 계산 => 평균 수익률 - 평균 국채 수익률 평균(수익률-국채수익률)
    final_sharpe = (mean_ret - mean_rtsy)/ (std_ret + 1e-8)
    
    return final_sharpe, decisions



# 1. 테스트 데이터 준비 
total_test_tensor = torch.tensor(total_test_data, dtype=torch.float32).to(device)

# 2. 테스트 데이터 분리
test_x_3d = total_test_tensor[:, :, :51]
test_y_3d = total_test_tensor[:, :, 51:]

# 3. 결과 도출 
sharpe_results, decisions = get_model_predictions(test_x_3d, test_y_3d, threshold=0.6)




print('working!!!')
