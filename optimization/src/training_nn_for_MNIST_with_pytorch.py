import numpy as np
from sympy import O
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys
from torchvision import datasets, transforms

#----------------------------------------------------------
# ハイパーパラメータなどの設定値
num_epochs = 5          # 学習を繰り返す回数
max_iterations = num_epochs*600        
num_batch = 100         # 一度に処理する画像の枚数
learning_rate = 0.01   # 学習率
image_size = 28*28      # 画像の画素数(幅x高さ)

# GPU(CUDA)が使えるかどうか
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#----------------------------------------------------------
# 学習用／評価用のデータセットの作成

# 変換方法の指定
transform = transforms.Compose([
    transforms.ToTensor()
    ])

# MNISTデータの取得
# 学習用
train_dataset = datasets.MNIST(
    './data',               # データの保存先
    train = True,           # 学習用データを取得する
    download = True,        # データが無い時にダウンロードする
    transform = transform   # テンソルへの変換など
    )
# 評価用
test_dataset = datasets.MNIST(
    './data', 
    train = False,
    transform = transform
    )

# データローダー
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = num_batch,
    shuffle = True)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,     
    batch_size = num_batch,
    shuffle = True)

#----------------------------------------------------------
# ニューラルネットワークモデルの定義
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(28*28, 100)
    self.fc2 = nn.Linear(100, 100)
    self.fc3 = nn.Linear(100, 100)
    self.fc4 = nn.Linear(100, 100)
    self.fc5 = nn.Linear(100, 10)
 
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    return self.fc5(x)
 
criterion = nn.CrossEntropyLoss()

#----------------------------------------------------------
# ニューラルネットワークの生成
models = {}
train_loss = {}
models['SGD'] = Net()
models['RMSprop'] = Net()
models['AdaGrad'] = Net()
models['Adam'] = Net()
models['AMSGrad'] = Net()
for key in models.keys():
    train_loss[key] = []

#----------------------------------------------------------
# 損失関数の設定
criterion = nn.CrossEntropyLoss() 

#----------------------------------------------------------
# 最適化手法の設定
optimizers = {}
optimizers['SGD'] = torch.optim.SGD(models['SGD'].parameters(),lr=learning_rate)
optimizers['RMSprop'] = torch.optim.RMSprop(models['RMSprop'].parameters(),lr=learning_rate)
optimizers['AdaGrad'] = torch.optim.Adagrad(models['AdaGrad'].parameters(),lr=learning_rate)
optimizers['Adam'] = torch.optim.Adam(models['Adam'].parameters(),lr=learning_rate)
optimizers['AMSGrad'] = torch.optim.Adam(models['AMSGrad'].parameters(),lr=learning_rate,amsgrad=True)

#----------------------------------------------------------
# 学習
for key in models.keys():
    models[key].train()  # モデルを訓練モードにする

loss_sums = {}
outputs = {}
losses = {}
loss_sums = {}
for epoch in range(num_epochs): # 学習を繰り返し行う
    for key in optimizers.keys():
        loss_sums[key] = 0

    for inputs, labels in train_dataloader:

        # GPUが使えるならGPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        # optimizerを初期化
        for key in optimizers.keys():
            optimizers[key].zero_grad()

        # ニューラルネットワークの処理を行う
        inputs = inputs.view(-1, image_size) # 画像データ部分を一次元へ並び変える
        for key in models.keys():
            outputs[key] = models[key](inputs)

        # 損失(出力とラベルとの誤差)の計算
        for key in outputs.keys():
            losses[key] = criterion(outputs[key], labels)
            train_loss[key].append(losses[key])
            
        for key in losses.keys():
            loss_sums[key] += losses[key]

        # 勾配の計算
        for key in losses.keys():
            losses[key].backward()

        # 重みの更新
        for key in optimizers.keys():
            optimizers[key].step()

    # 学習状況の表示
    for key in loss_sums.keys():
        print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss_sums[key].item() / len(train_dataloader)}")

    print("------------------------------------------------")

    # モデルの重みの保存
    for key in models.keys():
        torch.save(models[key].state_dict(), 'model_weights.pth')

#----------------------------------------------------------
# グラフの描画
def smooth_curve(x):
    window_len = 11
    with torch.no_grad():
        s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]

markers = {"SGD": "o", "RMSprop": ".", "AdaGrad": "s", "Adam": "D","AMSGrad": "x"}
x = np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 3)
plt.legend()
plt.show()

#----------------------------------------------------------
# 評価
loss_sums = {}
corrects = {}
outputs = {}
preds = {}

for key in models.keys():
    models[key].eval 

for key in models.keys():
    loss_sums[key] = 0

for key in models.keys():
    corrects[key] = 0

with torch.no_grad():
    for inputs, labels in test_dataloader:

        # GPUが使えるならGPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        # ニューラルネットワークの処理を行う
        inputs = inputs.view(-1, image_size) # 画像データ部分を一次元へ並び変える
        for key in models.keys():
            outputs[key] = models[key](inputs)

        # 損失(出力とラベルとの誤差)の計算
        for key in outputs.keys():
            loss_sums[key] += criterion(outputs[key],labels)

        # 正解の値を取得
        for key in outputs.keys():
            preds[key] = outputs[key].argmax(1)

        # 正解数をカウント
        for key in preds.keys():
            corrects[key] += preds[key].eq(labels.view_as(preds[key])).sum().item()
        
for key in models.keys():
    print(f"Loss: {loss_sums[key].item() / len(test_dataloader)}, Accuracy: {100*corrects[key]/len(test_dataset)}% ({corrects[key]}/{len(test_dataset)})")
