import torch
from PIL import Image
from model import FaceCNN  # 必须与训练时的模型定义一致
from torchvision import transforms
import json
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

def load_model():
    num_classes = count_files_in_folder('data/predict')  # 必须与训练时的类别数量一致
    model = FaceCNN(num_classes)
    model.load_state_dict(torch.load('face_model.pth'))
    model.eval()  # 设置为评估模式
    return model

#predict 函数用于对单张图像进行推理。
def predict(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 添加批次维度
    with torch.no_grad():
        output = model(image)
    return output.argmax().item()

def main_predict(predict_image_path):
    model = load_model()
    result = predict(predict_image_path, model)

    # 映射字典
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    predicted_label = class_names[result]

    # 展示结果
    return predicted_label

if __name__ == "__main__":
    model = load_model()
    test_image_path = './data/predict/李陈鑫.jpg'
    #展示结果
    print(f"预测结果: {main_predict(test_image_path)}")