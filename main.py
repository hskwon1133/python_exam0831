#캘리포니아 집값 예측
from matplotlib import pyplot as plt
from matplotlib.scale import scale_factory
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

plt.rcParams['font.family'] = ['Malgun Gothic']
plt.rcParams['axes.unicode_minus'] = False

X, y = fetch_california_housing(as_frame=True, return_X_y=True)
# X = data.data
# y = data.target

# print(X.head())
# print(y.head())

# 데이터 csv 저장하기
df = pd.concat([X,y], axis=1)
df.to_csv('california_housing.csv', index=False)

# 데이터 정보 확인
print(df.shape) # (20640, 9)
print(df.describe())
# print(df.describe()[['MedInc',  'HouseAge', 'AveRooms']]) # MinMax편차 큰 컬럼
# print(df.describe()[['AveBedrms',  'Population', 'AveOccup']])   # MinMax편차 큰 컬럼
# print(df.info()) # 9개 컬럼

'''
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   MedInc       20640 non-null  float64 중위소득(단위: 만달러)
 1   HouseAge     20640 non-null  float64 주택 연식
 2   AveRooms     20640 non-null  float64 가구당 평균 방개수
 3   AveBedrms    20640 non-null  float64 가구당 평균 침실 개수
 4   Population   20640 non-null  float64 해당 구역의 인구수
 5   AveOccup     20640 non-null  float64 가구당 평균 거주 인원 수
 6   Latitude     20640 non-null  float64 위도
 7   Longitude    20640 non-null  float64 경도
 8   MedHouseVal  20640 non-null  float64 중위 주택가격(단위: 10만 달러)
'''

# 결측치 확인
print(df.isna().sum()[df.isna().sum()>0]) #결측 X
# print(df.duplicated().sum()) # 중복 X
# print(df.min()==0) # 0의 결측치 확인 : None

# 데이터 탐색
plt.figure(figsize=(12,10))
sns.histplot(data = df['MedHouseVal'], bins=50)
plt.show()

# plt.figure(figsize=(12,10))
# cor = df.corr()
# sns.heatmap(data = cor, annot=True, cmap="coolwarm")
# plt.savefig('MedHouseVal_heatmap.png')
'''
소득에 따라 집값 가격의 영향력이 큼, 근소하게 연식, 평균 방수 또한 가격의 양수 영향력임.
반대로 위도의 위치에 따라 집값 음수 영항역이 근소하게 발생. 차이가 거의 없은 것은 평균 침실수 > 경도 > 평균 거주인원수 > 평균 인구수
'''

# 데이터 학습 준비

# 회귀분석 시 노의미
# scale_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
# scaler = StandardScaler().set_output(transform='pandas')
# X_scaled = scaler.fit_transform(df[scale_features])
# X_nonscale = df[['Latitude', 'Longitude']]
# X = pd.concat([X_scaled, X_nonscale], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print('y_train 범위:', y_train.min(), y_train.max())

scaler = StandardScaler() #아래 회귀분석 상 의미 없음. 문제가 있어 적용함.
m = LinearRegression()
pipe = make_pipeline(scaler, m)

#모델 학습
pipe.fit(X_train, y_train)

#예측
y_pred = pipe.predict(X_test)

#평가
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print('---Linear Regression---')
print('R2 score:', r2)
print('MAE score:', mae)
print('RMSE score:', rmse)
print('std: ', y.std())


'''
1차  Linear Regression  : 
R2 score: 0.5757877060324508
MAE score: 0.5332001304956565
RMSE score: 0.7455813830127764
std:  1.1539561587441385
'''

# x_train1, x_test1, y_train1, y_test1 = train_test_split(X, y, test_size = 0.2, random_state = 42)
# m = RandomForestRegressor(n_estimators=100, random_state=42)
# m.fit(x_train1, y_train1)
# y_pred1 = m.predict(x_test1)
# r2_1 = r2_score(y_test1, y_pred1)
# mae_1 = mean_absolute_error(y_test1, y_pred1)
# rmse_1 = np.sqrt(mean_squared_error(y_test1, y_pred1))
# print('---RandomForestRegressor---')
#
# print('R2 score:', r2_1)
# print('MAE score:', mae_1)
# print('RMSE score:', rmse_1)
# print('std: ', y.std())

'''
2차  RandomForestRegressor  : (위도 경도 = 비선형임)
R2 score: 0.8046244867176197
MAE score: 0.32773108008720936
RMSE score: 0.5059859946022769
std:  1.1539561587441385
'''