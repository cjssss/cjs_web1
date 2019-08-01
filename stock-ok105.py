import pandas as pd
import numpy as np
import requests
from io import BytesIO


from datetime import datetime, timedelta

def get_daily_price(code, fromdate=None, todate=None):
    if todate == None:
        todate = datetime.today().strftime('%Y%m%d')   # 오늘 날짜

    if fromdate == None:
        fromdate = (datetime.today() - timedelta(days=3650)).strftime('%Y%m%d')   # 30일 이전 날짜

    # STEP 01: Generate OTP
    gen_otp_url = "http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx"
    gen_otp_data = {
        'name':'fileDown',
        'filetype':'csv',
        'url':'MKD/04/0402/04020100/mkd04020100t3_02',
        'isu_cd':code,
        'fromdate':fromdate,
        'todate':todate,
    }
    
    r = requests.post(gen_otp_url, gen_otp_data)
    code = r.content  # 리턴받은 값을 아래 요청의 입력으로 사용.
    
    # STEP 02: download
    down_url = 'http://file.krx.co.kr/download.jspx'
    headers = {
        'Referer': 'http://marketdata.krx.co.kr/mdi'
    }    
    down_data = {
        'code': code,
    }
    
    r = requests.post(down_url, down_data, headers=headers)
    r.encoding = "utf-8-sig"
    df = pd.read_csv(BytesIO(r.content), header=0, thousands=',')

    return df

stock_no = 'KR7263800005'
df = get_daily_price(stock_no)

file_dir = 'c:\\users\\cjs\\ccc\\'
file_name = 'stock' + '.csv'

   
df = df.loc[:, ['년/월/일','시가','고가','저가','종가','종가','거래량(주)']]
df.columns= ['Date','Open','High','Low','Close','Adj close', 'Volume']
df = df.sort_values(['Date'], ascending=True)
df.to_csv(file_dir + file_name, index= False, index_label= None)
df.set_index('Date', inplace = True)

print(df.head())

import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

tf.set_random_seed(777)
 
def data_standardization(x): # Standardization
    x_np = np.asarray(x)
    return (x_np - x_np.mean()) / x_np.std()
 
def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7) # 1e-7은 0으로 나누는 오류 예방차원
 
def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()
 
 
# 하이퍼파라미터
input_data_column_cnt = 6  # 입력데이터의 컬럼 개수(Variable 개수)
output_data_column_cnt = 1 # 결과데이터의 컬럼 개수
 
seq_length = 28            # 1개 시퀀스의 길이(시계열데이터 입력 개수)
rnn_cell_hidden_dim = 20   # 각 셀의 (hidden)출력 크기
forget_bias = 1.0          # 망각편향(기본값 1.0)
num_stacked_layers = 1     # stacked LSTM layers 개수
keep_prob = 1.0            # dropout할 때 keep할 비율
 
epoch_num = 1000           # 에폭 횟수(학습용전체데이터를 몇 회 반복해서 학습할 것인가 입력)
learning_rate = 0.01       # 학습률
 
import pandas_datareader as pdr

# 데이터를 로딩한다.
stock_name = 'stock.csv' # 아마존 주가데이터 파일
encoding = 'euc-kr' # 문자 인코딩
names = ['Date','Open','High','Low','Close','Adj Close','Volume']
raw_dataframe = pd.read_csv(stock_name, names=names, encoding=encoding) #판다스이용 csv파일 로딩
raw_dataframe.info() # 데이터 정보 출력
 
# raw_dataframe.drop('Date', axis=1, inplace=True) # 시간열을 제거하고 dataframe 재생성하지 않기
del raw_dataframe['Date'] # 위 줄과 같은 효과
 
stock_info = raw_dataframe.values[1:].astype(np.float) # 금액&거래량 문자열을 부동소수점형으로 변환한다

price = stock_info[:,:-1]
norm_price = min_max_scaling(price) # 가격형태 데이터 정규화 처리

volume = stock_info[:,-1:]
norm_volume = min_max_scaling(volume) # 거래량형태 데이터 정규화 처리

x = np.concatenate((norm_price, norm_volume), axis=1) # axis=1, 세로로 합친다
 
y = x[:, [-2]] # 타켓은 주식 종가이다
print("y[0]: ",y[0])     # y의 첫 값
print("y[-1]: ",y[-1])   # y의 마지막 값
 
 
dataX = [] # 입력으로 사용될 Sequence Data
dataY = [] # 출력(타켓)으로 사용
 
for i in range(0, len(y) - seq_length):
    _x = x[i : i+seq_length]
    _y = y[i + seq_length] # 다음 나타날 주가(정답)
    #if i is 0:
        #print(_x, "->", _y) # 첫번째 행만 출력해 봄
    dataX.append(_x) # dataX 리스트에 추가
    dataY.append(_y) # dataY 리스트에 추가
 
train_size = int(len(dataY) * 0.7)
# 나머지(30%)를 테스트용 데이터로 사용
test_size = len(dataY) - train_size
 
# 데이터를 잘라 학습용 데이터 생성
trainX = np.array(dataX[0:train_size])
trainY = np.array(dataY[0:train_size])
 
# 데이터를 잘라 테스트용 데이터 생성
testX = np.array(dataX[train_size:len(dataX)])
testY = np.array(dataY[train_size:len(dataY)])
 
X = tf.placeholder(tf.float32, [None, seq_length, input_data_column_cnt])
print("X: ", X)
Y = tf.placeholder(tf.float32, [None, 1])
print("Y: ", Y)
 
# 검증용 측정지표를 산출하기 위한 targets, predictions를 생성한다
targets = tf.placeholder(tf.float32, [None, 1])
print("targets: ", targets)
 
predictions = tf.placeholder(tf.float32, [None, 1])
print("predictions: ", predictions)
 
# 모델(LSTM 네트워크) 생성
def lstm_cell():
    cell = tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', 
                        num_units=rnn_cell_hidden_dim, 
                        forget_bias=forget_bias, state_is_tuple=True, 
                        activation=tf.nn.softsign)
    
    if keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, 
                        output_keep_prob=keep_prob)
    return cell
 
# num_stacked_layers개의 층으로 쌓인 Stacked RNNs 생성
stackedRNNs = [lstm_cell() for _ in range(num_stacked_layers)]
multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True) if num_stacked_layers > 1 else lstm_cell()
 
# RNN Cell(여기서는 LSTM셀임)들을 연결
hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
print("hypothesis: ", hypothesis)
 
hypothesis = tf.contrib.layers.fully_connected(hypothesis[:, -1], 
                output_data_column_cnt, activation_fn=tf.identity)
 
loss = tf.reduce_sum(tf.square(hypothesis - Y))
# 최적화함수로 AdamOptimizer를 사용한다
optimizer = tf.train.AdamOptimizer(learning_rate)
 
train = optimizer.minimize(loss)
 
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets-predictions))) # 아래 코드와 같다
#rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))
 
train_error_summary = [] # 학습용 데이터의 오류를 중간 중간 기록한다
test_error_summary = []  # 테스트용 데이터의 오류를 중간 중간 기록한다
test_predict = ''        # 테스트용데이터로 예측한 결과
 
sess = tf.Session()
sess.run(tf.global_variables_initializer())
 
# 학습한다
start_time = datetime.datetime.now() # 시작시간을 기록한다
print('학습을 시작합니다...')
for epoch in range(epoch_num):
    _, _loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
    if ((epoch+1) % 100 == 0) or (epoch == epoch_num-1): # 100번째마다 또는 마지막 epoch인 경우
        # 학습용데이터로 rmse오차를 구한다
        train_predict = sess.run(hypothesis, feed_dict={X: trainX})
        train_error = sess.run(rmse, feed_dict={targets: trainY, predictions: train_predict})
        train_error_summary.append(train_error)
 
        # 테스트용데이터로 rmse오차를 구한다
        test_predict = sess.run(hypothesis, feed_dict={X: testX})
        test_error = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
        test_error_summary.append(test_error)
        
        # 현재 오류를 출력한다
        print("epoch: {}, train_error(A): {}, test_error(B): {}, B-A: {}".format(epoch+1, train_error, test_error, test_error-train_error))
        
end_time = datetime.datetime.now() # 종료시간을 기록한다
elapsed_time = end_time - start_time # 경과시간을 구한다
 
#plt.figure(2)
plt.figure(figsize=(12, 6))
plt.plot(testY, 'r', label="real")
plt.plot(test_predict, 'b', label="forecast")
plt.xlabel('Time Period')
plt.ylabel('Stock Price')
plt.grid()
plt.legend()
plt.show()
 
# sequence length만큼의 가장 최근 데이터를 슬라이싱한다
recent_data = np.array([x[len(x)-seq_length : ]])
 
# 내일 종가를 예측해본다
test_predict = sess.run(hypothesis, feed_dict={X: recent_data})
 
test_predict = reverse_min_max_scaling(price,test_predict) # 금액데이터 역정규화한다
print(stock_no +"  Tomorrow's price", test_predict[0]) # 예측한 주가를 출력한다