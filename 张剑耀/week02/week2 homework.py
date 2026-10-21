# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# week2 homework
# 任务：任意五维向量 x = [x0, x1, x2, x3, x4]，最大值出现在第几维，它就属于第几类（5 分类）
# 例：x = [0.1, 0.7, 0.3, 0.2, 0.5] -> 最大值 0.7 在第 1 维 -> 类别 1
#
# 学习方法：
#   1. 线性层输出 5 个 logits（未归一化的分数）
#   2. softmax 把 logits 变成概率：p_j = exp(z_j) / Σ exp(z_k)
#   3. 交叉熵衡量"预测概率分布"和"真实类别"的差距：L = -log(p_{真实类别})
#   4. 反向传播更新参数，让真实类别对应的概率越来越大



class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        # 输入 5 维，输出 5 个类别的 logits
        self.linear = nn.Linear(input_size, input_size)
        # 注意：这里不要写 softmax！
        # nn.functional.cross_entropy 内部已经做了 log_softmax，要求传入的是"未归一化的 logits"。
        # 如果自己先 softmax 再喂给 cross_entropy，等于做了两次归一化，梯度会变小、收敛变慢且数值不稳定。
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        logits = self.linear(x)          # (batch_size, 5)，未归一化的分数
        if y is not None:
            # y 是类别编号，形状 (batch_size,)，dtype 必须是 long
            return self.loss(logits, y)
        else:
            return logits

    def predict(self, x):
        """推理：手动做一次 softmax，把 logits 变成概率后再取最大"""
        self.eval()
        with torch.no_grad():
            prob = torch.softmax(self.linear(x), dim=1)  # 每行和为 1 dim 按行操作
            return torch.argmax(prob, dim=1), prob


def build_sample():
    """生成一个样本：五维随机向量 + 最大值所在的维度编号"""
    x = np.random.random(5)
    y = int(np.argmax(x))     # argmax 直接给出最大值下标，就是类别号（0~4）
    return x, y


def build_dataset(total_sample_num):
    """构造数据集。注意 Y 用 LongTensor 且是一维，cross_entropy 要求的就是这种格式"""
    X, Y = [], []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # 先转成 np.array 再转 Tensor，比直接传 list of ndarray 快得多（否则会收到一堆警告）ranho
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))


def evaluate(model):
    """用一批新数据测试准确率"""
    model.eval()
    test_sample_num = 200
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred, _ = model.predict(x)          # 预测类别
        for y_p, y_t in zip(y_pred, y):
            if int(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("测试样本 %d 个，正确 %d 个，正确率：%.4f" % (test_sample_num, correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def demo_softmax_crossentropy(model):
    """手工算一遍 softmax 和交叉熵，验证与 torch 的结果一致"""
    x, y = build_dataset(4)
    logits = model.forward(x)                       # (4, 5) 未归一化分数
    exp_z = torch.exp(logits)
    softmax_hand = exp_z / torch.sum(exp_z, dim=1, keepdim=True)   # 手写 softmax
    prob_torch = torch.softmax(logits, dim=1)                      # torch 的 softmax
    print("手写 softmax 与 torch.softmax 的最大差异: %.2e"
          % torch.max(torch.abs(softmax_hand - prob_torch)).item())

    # 交叉熵：真实类别是 k，则 L = -log(p_k)，再对所有样本取平均
    ce_hand = torch.mean(-torch.log(softmax_hand[torch.arange(len(y)), y]))
    ce_torch = nn.functional.cross_entropy(logits, y)
    print("手写交叉熵: %.6f   torch 交叉熵: %.6f   差异: %.2e"
          % (ce_hand.item(), ce_torch.item(), abs(ce_hand.item() - ce_torch.item())))


def main():
    epoch_num = 50
    batch_size = 32
    train_sample = 6000
    input_size = 5
    learning_rate = 0.01

    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    train_x, train_y = build_dataset(train_sample)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)     # 前向 + cross_entropy
            loss.backward()        # 反向传播
            optim.step()           # 更新参数
            optim.zero_grad()      # 清空梯度
            watch_loss.append(loss.item())

        acc = evaluate(model)
        avg_loss = float(np.mean(watch_loss))
        print("=========\n第%d轮平均loss: %.6f  正确率: %.4f" % (epoch + 1, avg_loss, acc))
        log.append([acc, avg_loss])

    torch.save(model.state_dict(), "model.bin")

    # 打印学到的权重：应该接近"放大版的单位阵"，第 j 类的第 j 维权重最大
    W = model.linear.weight.detach().numpy()
    print("学到的权重矩阵 W（行=类别，列=输入维度）：")
    print(np.round(W, 2))
    print("偏置 b:", np.round(model.linear.bias.detach().numpy(), 2))

    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.savefig("week2_training_curve.png")
    plt.show()
    return model


def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(input_vec)
        pred, prob = model.predict(x)
        for vec, p, pr in zip(input_vec, pred, prob):
            print("输入：%s\n  预测类别：%d  真实类别(argmax)：%d  各类概率：%s"
                  % (vec, int(p), int(np.argmax(vec)), np.round(pr.numpy(), 3)))


if __name__ == "__main__":
    trained_model = main()
    test_vec = [[0.81889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.39349776, 0.59416669, 0.92579291, 0.41567412, 0.8358894],
                [0.12345678, 0.98765432, 0.11111111, 0.22222222, 0.33333333]]
    predict("model.bin", test_vec)
    demo_softmax_crossentropy(trained_model)
