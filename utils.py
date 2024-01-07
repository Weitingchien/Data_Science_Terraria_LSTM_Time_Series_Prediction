import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def high_comments_count():
    file_path = 'daily_comment_count.txt'
    high_comment_counts = []  # This will store dates with comment counts of four or more digits

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()  # Strip to remove whitespace and split the line into parts
            # We expect two parts: date and count. If not, we skip the line
            if len(parts) == 2:
                date, count = parts
                print(count)
                if len(count) >= 4:  # Check if the count is four digits or more
                    high_comment_counts.append(f"{date} {count}\n")

    # Write the high comment counts to a new file
    new_file_path = 'high_comment_counts.txt'
    with open(new_file_path, 'w') as new_file:
        new_file.writelines(high_comment_counts)



def plot_comment_counts():
    file_path = 'high_comment_counts.txt'
    data = pd.read_csv(file_path, sep=' ', header=None, names=['Date', 'CommentCount'], skipfooter=1) # skipfooter=1 to skip the last line
    print(f"CommentCount: {data['CommentCount']}")
    data['Date'] = pd.to_datetime(data['Date'])
    plt.figure(figsize=(10,6))
    plt.bar(data['Date'], data['CommentCount'], color='skyblue')
    plt.xlabel('Date')
    plt.ylabel('Comment Count')
    plt.title('Comment Counts by Date')


    locator = mdates.AutoDateLocator(minticks=2, maxticks=5) # AutoDateLocator 會嘗試找到一個合理的方法來顯示這些日期標籤，既不過於擁擠，也不過於稀疏
    formatter = mdates.DateFormatter('%Y-%m-%d')  # 設定日期格式
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gcf().autofmt_xdate()  # 自動旋轉日期標記
    plt.tight_layout()

    plt.show()




def print_model_summary(model):
    for layer in model.named_modules():
        print(f"Layer: {layer[0]}")
        for param in layer[1].parameters():
            print(f"  - Parameter shape: {param.size()}")