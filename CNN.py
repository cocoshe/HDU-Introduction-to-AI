from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from utils.Vit import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # nn.Conv2d() 表示一个卷积层
        self.conv1 = nn.Sequential(
            # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # 输入的通道数
                out_channels=16,  # 过滤器的个数， 输出的通道数
                kernel_size=(5, 5),  # 卷积核的大小
                stride=(1, 1),  # 步长 filter movement/step
                padding=(2, 2)  # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # 是一个非线性激活函数
            nn.MaxPool2d(kernel_size=2),  # 池化层，在 2x2 空间里向下采样,
            # output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(
            # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),
            # output shape (32, 7, 7)
        )

        self.conv3 = nn.Sequential(
            # input shape (16, 14, 14)
            nn.Conv2d(32, 64, 5, 1, 2),  # output shape (64, 7, 7)
            nn.ReLU(),  # activation
            # output shape (64, 7, 7)
        )

        self.conv4 = nn.Sequential(
            # input shape (16, 14, 14)
            nn.Conv2d(64, 32, 5, 1, 2),  # output shape (32, 7, 7)
            nn.ReLU(),  # activation
            # output shape (32, 7, 7)
        )
        self.conv = nn.Conv2d(64, 32, 1)  # for concat
        # nn.Linear() 全连接层 ，一般作为输出层，再经过softmax就可以得到分类概率
        self.out = nn.Linear(32 * 7 * 7, 10)  # 全连接层，输出10个类别的预测

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x0 = x
        x = self.conv3(x)
        x = self.conv4(x)
        x = x + x0  # residual connection 1
        x = torch.concat([x, x0], 1)  # residual connection 2
        x = self.conv(x)  # [bs, 64, 7, 7] -> [bs, 32, 7, 7]
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output


# cnn = Net()
# print(cnn)  # net architecture


def train(args, model, device, train_loader, optimizer, epoch):
    """
    该函数的作用: 训练网络模型
        args: 参数对象
        model: 网络模型
        device: CPU or GPU
        train_loader： 加载训练图片
        optimizer： 优化器
        epoch：训练次数
    """

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        loss_func = nn.CrossEntropyLoss()  # 损失函数
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 清空前一次的梯度
        output = model(data)  # 将一个batch的数据，即tensor维度为(64, 1, 28, 28)传入神经网络模型
        loss = loss_func(output, target)  # 使用CrossEntropyLoss计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新神经网络的参数
        if batch_idx % args.log_interval == 0:  # 每10个batch，打印一次损失loss
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    """
    该函数的作用: 测试网络模型
        args: 参数对象
        model: 网络模型
        test_loader：加载测试图片
    """

    model.eval()  # 让 model 进入测试模式，不能省略
    test_loss = 0
    correct = 0
    with torch.no_grad():  # test神经网络，不进行梯度回传更新network
        for data, target in test_loader:
            loss_func = nn.CrossEntropyLoss()  # 损失函数
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--model', type=str, default="cnn",
                        help='choose model to run(default) cnn')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)   # 随机数种子reproducible

    device = torch.device("cuda" if use_cuda else "cpu")  # 选择用CPU训练还是GPU

    train_kwargs = {'batch_size': args.batch_size}  # 批训练的batch大小
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # 数据处理，转换tensor， 归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    '''
     'data' 为数据的存放路径
    如果你已经下载好了mnist数据，download就写上 False
    训练集部分的数据为dataset1，train=True，
    测试集为dataset2，故train=False,
    '''
    dataset1 = datasets.MNIST('data', train=True, download=False, transform=transform)
    dataset2 = datasets.MNIST('data', train=False, download=False, transform=transform)

    # 批训练 64 samples, 1 channel, 28x28 (64, 1, 28, 28)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if args.model == "cnn":
        model = Net().to(device)  # 神经网络模型初始化为model
    else:
        model = Vit(in_channels=1, embed_dim=64, depth=3).to(device)

    print("running: ", args.model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 优化器的选择

    # training and testing
    for epoch in range(1, args.epochs + 1):  # 训练轮次epoch=5
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    # 最后我们从测试集中取200个数据，再来取其中20个数据, 看看预测的值到底对不对
    test_x = torch.unsqueeze(dataset2.data, dim=1).type(torch.FloatTensor)[
             :200] / 255.  # shape from (200, 28, 28) to (200, 1, 28, 28), value in range(0,1)
    test_y = dataset2.targets[:200]

    test_output = model(test_x[:20])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:20].numpy(), 'real number')


if __name__ == '__main__':
    main()  # 执行入口函数
