import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# 데이터 칼럼명 수정


df = pd.read_csv("data_115m.csv") 
print(df.columns)


# 1. 실제 데이터 컬럼명에 맞게 명단 생성
# 0th_2th_prob, 2th_4th_prob, ..., 98th_100th_prob (총 50개)
prob_cols = [f'{i}th_{i+2}th_prob' for i in range(0, 100, 2)]

# 특성(X)으로 쓸 51개: 확률분포 50개 + 국채수익률 1개
feature_cols = prob_cols + ['Bond1']

# 타겟(Y)으로 쓸 1개: 실제 수익률
target_col = ['Return']

# 전체 사용할 컬럼 통합
all_cols = feature_cols + target_col

# 2. 원본 df에서 해당 컬럼들만 추출하여 복사본 생성
# 24GB 램을 활용해 안전하게 .copy()로 복제합니다.
df_copy = df[all_cols].copy()

# 3. 혹시 모를 결측치(NaN) 0으로 채우기
df_copy = df_copy.fillna(0)


feature_cols = [col for col in df_copy.columns if col != 'Return']



# 정규화
# 2. 스케일링 적용
scaler = StandardScaler()
df_copy[feature_cols] = scaler.fit_transform(df_copy[feature_cols])



# 결과 확인용 출력
print("✅ df_copy 생성 완료!")
print(f"전체 컬럼 수: {len(df_copy.columns)}개 (X: 51, Y: 1)")
print(f"데이터 크기: {df_copy.shape}")




# 최적화 및 학습 (GPU 최적화 버전)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = nn.Sequential(
    nn.Linear(51, 64),
    nn.ReLU(), # 활성화 함수: 비선형성을 추가하여 복잡한 패턴 학습
    nn.Linear(64, 1), # 출력층: 64개 정보를 모아 최종적으로 1개의 '점수' 도출
    nn.Sigmoid() # 시그모이드: 점수를 0~1 사이의 '확률' 값으로 변환
).to(device)

###### 여기에 자비어 넣기

# 최적화 도구: Adam을 사용하며, 학습률(lr)은 0.001로 설정
optimizer = optim.Adam(model.parameters(), lr=0.001)

# data_x_3d: (100, 3000, 51) -> 수익률 확률 분포와 국채 수익률
# data_y_3d: (100, 3000, 1)  -> 실제 수익률

def train_step(data_x_3d, data_y_3d, num_epochs=150):
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

        loss_history = []
        loss_history.append(loss.item()) # 매 에폭의 Loss 저장
    
    return loss_history # 기록된 리스트 반환





# train data set 준비

# 1. 데이터 불러오기 
df = pd.read_csv("data_115m.csv") 

# 컬럼 순서 가정: group1...group50 (50개) + Bond1 (1개) + Return (1개) = 총 52개
feature_cols = [f'group{i}' for i in range(1, 51)] + ['Bond1']
target_col = ['Return']
all_cols = feature_cols + target_col

# 2. 랜덤 샘플링 설정
TOTAL_ROWS = len(df_copy)  # 전체 115만 행
SAMPLE_SIZE = 300000  # 뽑을 데이터 30만 개
SETS = 100            # 팀 수
PER_SET = 3000        # 팀당 인원

# 전체 인덱스에서 30만 개를 중복 없이(replace=False) 랜덤 추출
all_indices = np.arange(TOTAL_ROWS)
sampled_indices = np.random.choice(all_indices, size=SAMPLE_SIZE, replace=False)

# 3. 데이터 추출 및 3D 변환
# 데이터프레임에서 샘플링된 인덱스만 가져온 후 넘파이로 변환
sampled_np = df_copy.iloc[sampled_indices].values # (300000, 52)

# (100, 3000, 52)로 모양 변경 (Reshape)
total_data_3d = sampled_np.reshape(SETS, PER_SET, 52)

# 4. 학습용 텐서 생성 및 분리 
total_tensor = torch.tensor(total_data_3d, dtype=torch.float32).to(device)

# [X 데이터]: 0~50번 컬럼 (group1~50 + Bond1) -> 총 51개
data_x_3d = total_tensor[:, :, :51] 

# [Y 데이터]: 51번 컬럼 (Return) -> 총 1개
data_y_3d = total_tensor[:, :, 51:] 

print(f"X shape: {data_x_3d.shape}") # (100, 3000, 51)
print(f"Y shape: {data_y_3d.shape}") # (100, 3000, 1)












# test

# 1. 데이터를 펴고 모델을 통과시켜 복구하는 함수
def get_model_predictions(test_x_3d, test_y_3d, threshold=0.6):
    model.eval() # 평가모드
    with torch.no_grad(): # 기울기 계산 금지 - 메모리, 속도
        # (10, 3000, 51) -> (30000, 51)로 펴기
        x_flat = test_x_3d.view(-1, 51)
        
        # 모델 통과 (결과: 30000,)
        probs_flat = model(x_flat)
        
        # 원래 구조로 복구 (결과: 10, 3000)
        probs_3d = probs_flat.view(10, 3000)

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





# [과정] 115만 명 중 학습에 쓴 30만 명을 제외한 나머지에서 테스트용(예: 3만 명) 추출
# 1. 30만을 제외한 데이터 중 맨 앞부터 3만개만 추출
test_indices = np.setdiff1d(all_indices, sampled_indices)[:30000] 

# 2. 테스트 데이터 구성 (10팀, 3000명, 52컬럼)
test_np = df_copy.iloc[test_indices].values
total_test_data = test_np.reshape(10, 3000, 52) # 10세트 구성

total_test_tensor = torch.tensor(total_test_data, dtype=torch.float32).to(device)

# 3. 테스트 데이터 분리
test_x_3d = total_test_tensor[:, :, :51]
test_y_3d = total_test_tensor[:, :, 51:]





# 1. 모델 학습 실행
print("\n" + "="*50)
print(" [Step 1] 모델 학습(Training) 시작...")
print("="*50)
loss_history = train_step(data_x_3d, data_y_3d, num_epochs=300)
print(">> 학습 완료!")

# 2. 테스트 데이터 결과 도출
print("\n" + "="*50)
print(" [Step 2] 테스트 데이터(Test Data) 평가 시작...")
print("="*50)
sharpe_results, decisions = get_model_predictions(test_x_3d, test_y_3d, threshold=0.6)



# 3. 상세 수치 계산 (텐서를 넘파이로 변환하여 계산)
avg_sharpe = sharpe_results.mean().item()
max_sharpe = sharpe_results.max().item()
min_sharpe = sharpe_results.min().item()

total_test_count = decisions.numel()
approved_count = int(decisions.sum().item())
approval_rate = (approved_count / total_test_count) * 100
approved_returns = test_y_3d.cpu().numpy()[decisions == 1]

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
