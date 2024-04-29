import csv

def add_record_to_csv(filepath, data_dict):
    with open(filepath, 'a', newline='') as file:  # 'a' 模式用于追加数据
        writer = csv.DictWriter(file, fieldnames=data_dict.keys())
        if file.tell() == 0:  # 检查文件是否为空，如果为空则写入表头
            writer.writeheader()
        writer.writerow(data_dict)

# 使用示例
data1 = {'name': 'Alice', 'age': 30, 'city': 'New York'}
data2 = {'name': 'Alice2', 'age': 32, 'city': 'New York'}
add_record_to_csv('data.csv', data1)
add_record_to_csv('data.csv', data2)

