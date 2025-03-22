import os

def count_files_in_folder(folder_path):
    file_count = 0
    try:
        # 获取指定文件夹下的所有文件和子文件夹
        items = os.listdir(folder_path)
        for item in items:
            item_path = os.path.join(folder_path, item)
            # 判断是否为文件
            if os.path.isfile(item_path):
                file_count = file_count + 1
    except FileNotFoundError:
        print(f"错误：未找到文件夹 {folder_path}。")
    except Exception as e:
        print(f"发生未知错误：{e}")
    return file_count

# 示例使用
file_count = count_files_in_folder('data/predict')
print(f"文件夹 {folder_path} 中的文件数量为：{file_count}")