import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# 데이터 칼럼명 수정


df = pd.read_csv("distribution_model_MLP.csv")


# 1. 실제 데이터 컬럼명에 맞게 명단 생성
prob_cols = [f'{i}th_{i+2}th_prob' for i in range(0, 100, 2)]

# 특성(X)으로 쓸 51개: 확률분포 50개 + 국채수익률 1개
feature_cols = prob_cols + ['Bond']

# 타겟(Y)으로 쓸 1개: 실제 수익률
target_col = ['Return']

# 전체 사용할 컬럼 통합
all_cols = feature_cols + target_col

# 2. 원본 df에서 해당 컬럼들만 추출하여 복사본 생성
df_copy = df[all_cols].copy()

# 3. 혹시 모를 결측치(NaN) 0으로 채우기
df_copy = df_copy.fillna(0)


feature_cols = [col for col in df_copy.columns if col != 'Return']




#------------------------------------------------------------------------------

# 스케일링
scaler = StandardScaler()
df_copy[feature_cols] = scaler.fit_transform(df_copy[feature_cols])


# 최적화 및 학습 (GPU 최적화 버전)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Sequential(
    nn.Linear(51, 64),
    nn.ReLU(), # 활성화 함수: 비선형성을 추가하여 복잡한 패턴 학습
    nn.Linear(64, 64),
    nn.ReLU(), # 활성화 함수: 비선형성을 추가하여 복잡한 패턴 학습
    nn.Linear(64, 1), # 출력층: 64개 정보를 모아 최종적으로 1개의 '점수' 도출
    nn.Sigmoid() # 시그모이드: 점수를 0~1 사이의 '확률' 값으로 변환
).to(device)

###### 여기에 자비어 넣기

# 최적화 도구: Adam을 사용하며, 학습률(lr)은 0.001로 설정
optimizer = optim.Adam(model.parameters(), lr=0.001)








#------------------------------------------------------------------------------

# train data set 준비

# 컬럼 순서 가정: group1...group50 (50개) + Bond1 (1개) + Return (1개) = 총 52개
feature_cols = [f'group{i}' for i in range(1, 51)] + ['Bond1']
target_col = ['Return']
all_cols = feature_cols + target_col


# 0. 설정 값 정의
PER_SET = 10000
TRAIN_SIZE = 800000  # 80팀
TEST_SIZE = 200000   # 20팀

# 실제 팀 수 계산 
train_sets = TRAIN_SIZE // PER_SET  # 800팀
test_sets = TEST_SIZE // PER_SET

# 1. 인덱스 분리 (중복 방지)
all_indices = np.arange(len(df_copy))

# 학습용 추출
train_idx = np.random.choice(all_indices, size=TRAIN_SIZE, replace=False)

# 학습용 제외한 나머지
remaining_idx = np.setdiff1d(all_indices, train_idx)

test_idx = np.random.choice(remaining_idx, size=test_sets * PER_SET, replace=False)

# 2. 3D 텐서 변환 함수 (반복 작업 최적화)
def make_3d_tensor(indices, num_sets):
    # 데이터 추출 및 Reshape (팀수, 팀당인원, 컬럼수)
    data_np = df_copy.iloc[indices].values
    data_3d = data_np.reshape(num_sets, PER_SET, 52)
    
    # 텐서 변환 및 장치 이동
    tensor = torch.tensor(data_3d, dtype=torch.float32).to(device)
    
    # X(피처 51개)와 Y(타겟 1개) 분리
    x = tensor[:, :, :51]
    y = tensor[:, :, 51:]
    return x, y

# 3. 최종 데이터 생성
train_x, train_y = make_3d_tensor(train_idx, train_sets)
test_x, test_y   = make_3d_tensor(test_idx, test_sets)

# 결과 확인
print("✅")






#------------------------------------------------------------------------------

def train_step(data_x_3d, data_y_3d, num_epochs=150):
    for epoch in range(num_epochs):
        model.train()
        
        # 1. Forward
        x_flat = data_x_3d.view(-1, 51) # (80, 10000, 51) -> (800,000, 51)
        probs_flat = model(x_flat).squeeze(-1) # 모델 통과 후 승인확률 get 1차원 형태로
        probs_3d = probs_flat.view(80, 10000) # 다시 (80팀, 10000명)으로 복구
        
        # 2. 데이터 분리 
        r_tsy_3d = data_x_3d[:, :, -1] # 국채수익률만 분리
        actual_ret_3d = data_y_3d.squeeze(-1) # (80, 10000) 차원 일치
        
        # 3. probs_3d가 1에 가까워지면 actual_ret_3d가 되고, 0에 가까워지면 r_tsy_3d
        individual_returns = (probs_3d * actual_ret_3d) + ((1 - probs_3d) * r_tsy_3d)
        
        mean_ret = individual_returns.mean(dim=1) # dim=1 : 가로 방향. 한 팀 안에 있는 10,000명의 평균을 계산
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

        loss_history = []
        loss_history.append(loss.item()) # 매 에폭의 Loss 저장
    
    return loss_history # 기록된 리스트 반환














#------------------------------------------------------------------------------

# test

# 1. 데이터를 펴고 모델을 통과시켜 복구하는 함수
def get_model_predictions(test_x_3d, test_y_3d, threshold=0.6):
    model.eval() # 평가모드
    with torch.no_grad(): # 기울기 계산 금지 - 메모리, 속도
        # (10, 3000, 51) -> (30000, 51)로 펴기
        x_flat = test_x_3d.view(-1, 51)
        
        # 모델 통과 (결과: 200000,)
        probs_flat = model(x_flat)
        
        # 원래 구조로 복구 (결과: 20, 10000)
        probs_3d = probs_flat.view(-1, 10000)

        r_tsy_3d = test_x_3d[:, :, -1] # 국채 수익률 분리
        actual_ret_3d = test_y_3d.squeeze(-1) # 실제 수익률
    

        # 함수 내부나 호출 직후에 추가
        print(f"--- 모델 예측값 분석 ---")
        print(f"최댓값: {probs_3d.max().item():.4f}")
        print(f"최솟값: {probs_3d.min().item():.4f}")
        print(f"평균값: {probs_3d.mean().item():.4f}")

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






#------------------------------------------------------------------------------

# 1. 모델 학습 실행
print("\n" + "="*50)
print(" [Step 1] 모델 학습(Training) 시작...")
print("="*50)
loss_history = train_step(train_x, train_y, num_epochs=300)
print(">> 학습 완료!")

# 2. 테스트 데이터 결과 도출
print("\n" + "="*50)
print(" [Step 2] 테스트 데이터(Test Data) 평가 시작...")
print("="*50)
sharpe_results, decisions = get_model_predictions(test_x, test_y, threshold=0.6)



# 3. 상세 수치 계산 (텐서를 넘파이로 변환하여 계산)
avg_sharpe = sharpe_results.mean().item()
max_sharpe = sharpe_results.max().item()
min_sharpe = sharpe_results.min().item()

total_test_count = decisions.numel()
approved_count = int(decisions.sum().item())
approval_rate = (approved_count / total_test_count) * 100
approved_returns = test_y.cpu().numpy()[decisions == 1]

if len(approved_returns) > 0:
    avg_approved_return = approved_returns.mean() * 100  # % 단위로 변환
else:
    avg_approved_return = 0.0  # 승인된 대출이 하나도 없을 경우 대비

# 결과 출력
print("\n" + "*"*20 + " [ 최종 성적표 ] " + "*"*20)
print(f"▶ 전체 테스트 인원      : {total_test_count:,} 명")
print(f"▶ 모델이 승인한 인원    : {approved_count:,} 명")
print(f"▶ 최종 승인율          : {approval_rate:.2f} %")
print("-" * 51)
print(f"▶ 평균 샤프 지수(Sharpe) : {avg_sharpe:.4f}")
print(f"▶ 최고 팀 샤프 지수     : {max_sharpe:.4f}")
print(f"▶ 최저 팀 샤프 지수     : {min_sharpe:.4f}")
print("-" * 51)
print(f"▶ 승인된 대출의 평균 수익률 : {avg_approved_return:.2f} %")
print("*"*54)



