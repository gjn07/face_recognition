from model import train_model

def train_and_save(num_epochs):
    # ...（训练代码train_model 函数）
    train_model(num_epochs)

if __name__ == "__main__":
    # #num_epochs 表示训练的轮数。
    num_epochs = 15;
    train_and_save(num_epochs)