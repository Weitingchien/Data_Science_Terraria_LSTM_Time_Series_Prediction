import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import datetime
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from torch.utils.data import TensorDataset, DataLoader

from statsmodels.tsa.stattools import adfuller


from utils import high_comments_count



# 進行 ADF 檢驗
def plot_adf_test(series, adf_result):
    plt.figure(figsize=(14, 7), dpi=120)
    plt.plot(series, label='Time Series')
    
    # ADF Statistic
    adf_statistic = adf_result[0]
    plt.axhline(y=adf_statistic, color='r', linestyle='--', label=f'ADF Statistic = {adf_statistic:.2f}')
    
    # Critical Values
    for key, value in adf_result[4].items():
        plt.axhline(y=value, color='g', linestyle='--', label=f'Critical Value ({key}) = {value:.2f}')


    
    plt.title('ADF Test Result')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend(loc='best')  # 確保圖例在最佳位置
    plt.tight_layout()
    plt.show()






""" look_back=3表示使用過去3天預測第4天的值"""

"""
    假設我的data是[[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
    經過迴圈得到以下結果：
    X: [[10 20 30][20 30 40][30 40 50][40 50 60][50 60 70][60 70 80]]
    y: [40 50 60 70 80 90]
"""
def create_dataset(data, look_back=3): 
    # print(f'len(data): {len(data)}')
    X, y = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        X.append(a)
        y.append(data[i + look_back, 0])
    return np.array(X).reshape(-1, 1, look_back), np.array(y)




# 定義 LSTM 模型
class LSTMModel(nn.Module):
    # hidden_layer_size: 每層神經元的數量(預設設定為60個神經元)
    def __init__(self, input_size=1, hidden_layer_size=60, output_size=1, num_layers=2, num_directions=1, dropout_rate=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.num_directions = num_directions # num_directions = 1 表示單向LSTM
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) # 訓練期間用於正則化的丟棄率
        self.linear = nn.Linear(hidden_layer_size, output_size) # 從隱藏層大小映射到輸出大小
        self.hidden_cell = None

    # 初始化 LSTM 的隱藏狀態和單元狀態
    def init_hidden(self, batch_size):
        # This matches the number of layers and the directionality of the LSTM
        hidden = (torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_layer_size).to(next(self.parameters()).device),
                  torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_layer_size).to(next(self.parameters()).device))
        self.hidden_cell = hidden

    # input_seq 是輸入數據
    def forward(self, input_seq):
        batch_size = input_seq.size(0) # 從輸入序列獲取批次大小

        self.init_hidden(batch_size) # 為當前批次初始化隱藏狀態和單元狀態

        input_seq = input_seq.transpose(0, 1) # 轉置輸入序列, 以適應 LSTM 層的期望輸入格式, 格式為 (seq_len, batch, input_size)

        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell) # 將輸入序列和隱藏狀態傳遞給 LSTM, 獲取輸出和新的隱藏狀態
        lstm_out = self.dropout(lstm_out) 

        # 將 lstm_out 的形狀從 (seq_len, batch, hidden_layer_size) 更改為 (seq_len * batch, hidden_layer_size)
        # -1 告訴 PyTorch 自動計算這一維的大小
        predictions = self.linear(lstm_out.view(-1, self.hidden_layer_size))

        # 將預測值的形狀調整為 (batch_size, time_steps, output_size)
        predictions = predictions.view(batch_size, -1, self.output_size)

        # 取每個批次的最後一個時間步的輸出作為最終預測值
        return predictions[:, -1, :] # 為每個批次返回序列的最後一個預測, 對下一天的評論計數的預測


def groupby_timestamp():
    # 將 timestamp 轉換為日期
    data = read_csv_file()
    data['date'] = pd.to_datetime(data['timestamp_created'], unit='s').dt.date
    # 計算每天的評論數量
    daily_comment_count = data.groupby('date').size()
    return daily_comment_count


def read_csv_file():
    # 讀取用戶提供的 CSV 文件
    file_path = './dataset/review_105600.csv'
    data = pd.read_csv(file_path)

    
    return data




# 然後在訓練函數中迭代 DataLoader
def train_model(train_loader, val_loader, model, optimizer, loss_function, patience=5):
    best_loss = float('inf') # 初始值設置為無限大
    best_model = None
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    for epoch in range(300):
        model.train()
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq)
            train_loss = loss_function(y_pred, labels.view(-1, 1))
            train_loss.backward()
            optimizer.step()

        model.eval()
        val_loss_epoch = []
        with torch.no_grad():
            for seq, labels in val_loader:
                y_pred = model(seq)
                val_loss = loss_function(y_pred, labels.view(-1, 1))
                val_loss_epoch.append(val_loss.item())

        avg_train_loss = train_loss.item()
        avg_val_loss = np.mean(val_loss_epoch)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(f'Epoch {epoch}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model = model.state_dict()
            epochs_without_improvement = 0

             # 保存最佳模型的參數
            torch.save(best_model, 'best_model.pth')
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print('Early stopping triggered')
            break

    model.load_state_dict(best_model)
    return model, train_losses, val_losses





def plot_original_data_predicted_data(test_predictions, test_labels, y_test):
    # 假設 test_dates 包含對應於 y_test 中每個點的日期
    daily_comment_count = groupby_timestamp()
    test_dates = daily_comment_count.index[-len(y_test):]
    # print(f'test_dates: {test_dates} test_dates(len):{len(test_dates)}')

    # print(f'test_labels:{test_labels} test_labels(len): {len(test_labels)}')

    # 確保日期和數據都是正確對齊的
    assert len(test_dates) == len(test_labels)

    # 轉換日期為 matplotlib 能理解的格式
    dates = [mdates.datestr2num(str(date)) for date in test_dates]
    test_dates = [date for date in test_dates]
    # print(f'test_dates: {test_dates}')

    # 繪製原始數據和預測數據
    plt.figure(figsize=(15, 6))
    plt.gca().set_xlim([min(test_dates), max(test_dates)])
    plt.plot_date(dates, test_labels, 'b-', label='Original Data')
    plt.plot_date(dates, test_predictions, 'r-', label='Predicted Data')

    # 美化圖表
    plt.title('Comparison of Original and Predicted Data')
    plt.xlabel('Date')
    plt.ylabel('Comment Count')
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_train_lossed_val_losses(train_losses, val_losses):
    # 繪製訓練損失和驗證損失
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Train vs Validation Loss')   
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()






def eval_model(test_loader, scaler, model, test_dates):
    model.eval()
    test_predictions = []
    test_labels_list = []  # 用來存放轉換後的 NumPy 陣列

    with torch.no_grad():
        for seq, labels in test_loader:
            model.init_hidden(seq.size(0))
            y_pred = model(seq)
            test_predictions.extend(y_pred.cpu().numpy())
            test_labels_list.append(labels.cpu().numpy())

    test_predictions = np.concatenate(test_predictions).flatten()
    test_labels = np.concatenate(test_labels_list).flatten()  # 拼接轉換後的標籤陣列

    test_predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1))
    test_labels = scaler.inverse_transform(test_labels.reshape(-1, 1))

    mse = np.mean((test_predictions - test_labels)**2)
    print(f"Test MSE: {mse}")

    rmse = np.sqrt(mse)
    print(f"Test RMSE: {rmse}")

      # 儲存預測結果到 predicted_comments.csv
    with open('predicted_comments.csv', 'w') as f:
        f.write('Date,Predicted Comment Count\n')
        for date, prediction in zip(test_dates, test_predictions):
            f.write(f"{date},{prediction[0]}\n")

    return test_predictions, test_labels





def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    high_comments_count()
    
    

    pd.set_option('display.max_rows', 4366)
    daily_comment_count = groupby_timestamp()
    result = adfuller(daily_comment_count.values, autolag='AIC')
    plot_adf_test(daily_comment_count, result)

    with open('daily_comment_count.txt', 'w') as f:
        print(daily_comment_count, file=f)



    look_back = 50 # 參數是用來決定在預測未來值時應該考慮多少過去的時間步
    """測試用
    scaled_data = pd.DataFrame([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).values.reshape(-1,1)
    create_dataset(scaled_data,look_back)
    """

    
    # 分割數據集
    train_size = int(len(daily_comment_count) * 0.8) #80%: 3489
    val_size = int(len(daily_comment_count) * 0.1) #10%: 437
    test_size = len(daily_comment_count) - train_size - val_size #10%: 437

    data_train = daily_comment_count[:train_size]
    data_val = daily_comment_count[train_size:train_size+val_size]
    data_test = daily_comment_count[train_size+val_size:]

    # 正規化(normalize), 讓值介於0~1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_train_scaled = scaler.fit_transform(data_train.values.reshape(-1, 1))
    data_val_scaled = scaler.transform(data_val.values.reshape(-1, 1))
    data_test_scaled = scaler.transform(data_test.values.reshape(-1, 1))


    X_train, y_train = create_dataset(data_train_scaled, look_back)
    X_val, y_val = create_dataset(data_val_scaled, look_back)
    X_test, y_test = create_dataset(data_test_scaled, look_back)


    model = LSTMModel(input_size=look_back, hidden_layer_size=80, output_size=1)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # 將數據轉換為適合 DataLoader 的形式
    train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)  # 批次大小為64

    # 創建validate DataLoader
    val_data = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    val_loader = DataLoader(dataset=val_data, batch_size=64, shuffle=False)

    # 創建測試 DataLoader
    test_data = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

    # 訓練模型
    model, train_losses, val_losses = train_model(train_loader, val_loader, model, optimizer, loss_function, patience=5)


    # 評估模型
    test_dates = daily_comment_count.index[-len(y_test):]  # 確保這些日期與 y_test 對應
    test_predictions, test_labels = eval_model(test_loader, scaler, model, test_dates)



    plot_original_data_predicted_data(test_predictions, test_labels, y_test)
    plot_train_lossed_val_losses(train_losses, val_losses)



if __name__ == "__main__":
    main()