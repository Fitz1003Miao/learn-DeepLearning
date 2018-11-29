import torch
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import os
import AlexNet
from PIL import Image
import argparse

CLASS_LIST = ['cat', 'dog']

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.data_path = [os.path.join(folder, file) for file in os.listdir(folder)]
        self.trans = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    def __getitem__(self, index):
        img_path = self.data_path[index]
        label = CLASS_LIST.index(os.path.basename(img_path).split('.')[0])
        img = self.trans(Image.open(img_path))
        return img, label
    def __len__(self):
        return len(self.data_path)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        import math
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def valmodel(model, dataloader, criterion, use_gpu):
    model.eval()

    total_loss = 0
    total_correct = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        
        loss = criterion(output, target)
        total_loss += loss.item()
        
        pred = torch.max(F.softmax(output), 1)[1]
        total_correct += (pred == target).sum().item()

    print("Val: Loss: {:.6f}, Acc: {:.6f}".format(total_loss / len(dataloader), total_correct / len(dataloader.dataset)))

def trainmodel(model, dataloader, criterion, optimizer, use_gpu):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        output = model(data)
        pred = torch.max(F.softmax(output), 1)[1]
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        print('Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccu: {:.6f}'.format(
                batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item(), (target == pred).sum().item() / target.size(0)))

def savemodel(model, optimizer, outpath):
    torch.save({'model_state_dict' : model.state_dict(), 'optimizer_state_dict' : optimizer.state_dict(), 'epoch' : outpath}, outpath)

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('-t', '--train', help = 'Path to train data')
    parse.add_argument('-v', '--val', help = 'Path to val data')
    parse.add_argument('-o', '--outpath', help = 'Path to save model')
    parse.add_argument('-m', '--model', help = 'Path to load model')

    args = parse.parse_args()
    print(args)

    train_dataset = MyDataSet(args.train)
    val_dataset = MyDataSet(args.val)

    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath, exist_ok = True)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 128, shuffle = True, num_workers = 0)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = 128, shuffle = False, num_workers = 0)

    model = AlexNet.AlexNet(num_classes = len(CLASS_LIST))
    model.apply(weight_init)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
        print('USE GPU')
    else:
        print('USE CPU')

    criterion = nn.CrossEntropyLoss(size_average = True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay = 0.0005)

    if args.model is not None:
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(checkpoint['epoch'])

    for epoch in range(1, 100):
        trainmodel(model, train_dataloader, criterion, optimizer, use_gpu)
        valmodel(model, val_dataloader, criterion, use_gpu)
        savemodel(model, optimizer, os.path.join(args.outpath, 'iter-{:04d}'.format(epoch)))
if __name__ == "__main__":
    main()