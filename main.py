import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from predict import  main_predict

# 假设这是你的人脸识别函数
def face_recognition(image_path):
    # 这里应该替换为你的实际人脸识别代码
    return f"识别结果： {main_predict(image_path)}"

def on_listbox_select(event):
    selected_index = listbox.curselection()
    if selected_index:
        # 重置识别结果标签的文本
        result_label.config(text="识别结果将显示在这里", fg="#333", font=("Arial", 11))

        selected_file = listbox.get(selected_index)
        file_path = os.path.join(folder_path, selected_file)
        try:
            # 显示图片（保持宽高比）
            img = Image.open(file_path)
            width, height = img.size
            max_size = 300  # 图片显示区域的限制大小
            ratio = min(max_size / width, max_size / height)
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            image_label.config(image=photo)
            image_label.image = photo
        except Exception as e:
            messagebox.showerror("错误", f"发生错误: {e}")

def select_image():
    selected_index = listbox.curselection()
    if selected_index:
        selected_file = listbox.get(selected_index)
        file_path = os.path.join(folder_path, selected_file)
        try:
            result = face_recognition(file_path)
            result_label.config(text=result, fg="blue", font=("Arial", 12))
        except Exception as e:
            messagebox.showerror("错误", f"发生错误: {e}")


# 选择文件夹
folder_path = filedialog.askdirectory()
if not folder_path:
    messagebox.showerror("错误", "未选择文件夹")
else:
    # 创建主窗口
    root = tk.Tk()
    root.title("人脸识别系统")
    root.geometry("800x600")  # 固定窗口大小

    # 添加标题标签
    title_label = tk.Label(root, text="人脸识别系统", font=("Arial", 24, "bold"), fg="darkblue")
    title_label.pack(pady=20)

    # 主布局分为左右两列
    left_frame = tk.Frame(root)
    left_frame.pack(side="left", padx=20, pady=20, fill="both")

    right_frame = tk.Frame(root)
    right_frame.pack(side="right", padx=20, pady=20, fill="both", expand=True)

    # 左侧区域：列表框和按钮
    listbox_label = tk.Label(left_frame, text="图片列表", font=("Arial", 12))
    listbox_label.pack(pady=5)

    listbox = tk.Listbox(left_frame, width=30, height=20)
    listbox.bind("<<ListboxSelect>>", on_listbox_select)

    # 添加滚动条
    scrollbar = tk.Scrollbar(left_frame)
    scrollbar.pack(side="right", fill="y")
    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)

    listbox.pack(pady=10)

    select_button = tk.Button(left_frame, text="进行识别", command=select_image,
                              bg="#4CAF50", fg="white", font=("Arial", 10))
    select_button.pack(pady=10, ipadx=10, ipady=5)

    # 右侧区域：图片和结果
    image_label = tk.Label(right_frame, bg="lightgray", bd=2, relief="groove")
    image_label.pack(pady=20)

    result_label = tk.Label(right_frame, text="识别结果将显示在这里",
                            wraplength=300, font=("Arial", 11), fg="#333")
    result_label.pack(pady=10)

    # 加载图片文件列表
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for file in image_files:
        listbox.insert(tk.END, file)

    root.mainloop()
