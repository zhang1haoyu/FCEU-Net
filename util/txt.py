import os
from tqdm import tqdm

def read_root(root):
    # Implement the function to read and return files from the root
    pass

def mean_std(files, h, w):
    # Implement the function to calculate mean and std
    pass

root = r'D:\Potsdam'
train_files = os.listdir(r'D:\Potsdam\train\img')
test_files = os.listdir(r'D:\Potsdam\test\img')

print("Train files:", train_files)  # 调试代码，打印文件列表
print("Test files:", test_files)    # 调试代码，打印文件列表

train_txt_path = os.path.join(root, 'train', 'train.txt')
test_txt_path = os.path.join(root, 'test', 'test.txt')

# 创建文件夹
os.makedirs(os.path.dirname(train_txt_path), exist_ok=True)
os.makedirs(os.path.dirname(test_txt_path), exist_ok=True)

# 写入训练文件
with open(train_txt_path, 'w') as train_txt:
    for file in tqdm(train_files):
        train_txt.write(file + '\n')

# 写入测试文件
with open(test_txt_path, 'w') as test_txt:
    for file in tqdm(test_files):
        test_txt.write(file + '\n')

print(mean_std(read_root(r'D:\Potsdam\train\img'), h=256, w=256))
