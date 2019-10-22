#!/usr/bin/env python3
import sys
import time
from tqdm import tqdm
import argparse
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as fn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import transforms, datasets

preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(28),
    transforms.CenterCrop(28),
    transforms.ToTensor(),
])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def train_model(testset, epochs=2, batch_size=64, no_cuda=False):
    training = datasets.MNIST("", train=True, download=True,
                              transform=transforms.Compose([transforms.ToTensor()]))

    trainset = torch.utils.data.DataLoader(
        training, batch_size=batch_size, shuffle=True)

    device = torch.device('cpu' if no_cuda else 'cuda')

    net = Net().to(device)
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print("Begin Epoch", epoch + 1, "/", epochs)
        for (data, target) in tqdm(trainset):
            data, target = data.to(device), target.to(device)
            net.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print('Loss:', loss.item())
        with torch.no_grad():
            count = 0
            count_matches = 0
            for (data, target) in testset:
                data = data.to(device)
                output = net(data)
                for i, output in enumerate(output):
                    if torch.argmax(output) == target[i]:
                        count_matches += 1
                    count += 1
            print('Accuracy:', count_matches * 100.0 / count)

    return net


def load_model(path):
    the_model = Net()
    the_model.load_state_dict(torch.load(path))
    return the_model


def evaluate(image_folder, model, no_cuda=False):
    device = torch.device('cpu' if no_cuda else 'cuda')
    model.to(device)
    model.eval()
    with torch.no_grad():
        size = len(image_folder)
        for idx, (data, target) in enumerate(image_folder):
            plt.subplot(1, size, idx + 1)
            plt.axis('off')
            plt.imshow(data.view(28, 28))
            data = data.to(device)
            output = model(data.view(-1, 1, 28, 28))
            plt.title(str(torch.argmax(output).item()))
        plt.savefig('output.png')
        plt.show()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Digits Example')
    parser.add_argument('--train', action='store_true', default=False,
                        help='train and save the model')
    parser.add_argument('--model', type=str, default="./models/numbers.model", metavar='PATH',
                        help='path to the trained model (default: ./models/numbers.model)')
    parser.add_argument('--images', type=str, default="./images/", metavar='PATH',
                        help='path to the testing data (default: ./images/)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='number of samples per batch (default: 64)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs (default: 2)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='use cpu for training and evaluation')                        

    args = parser.parse_args()

    if not (torch.cuda.is_available() or args.no_cuda):
        print('Err: no cuda device found, run with option --no-cuda', file=sys.stderr)
        exit(1)

    image_path = args.images

    image_folder = torchvision.datasets.ImageFolder(
        root=image_path,
        transform=preprocess,
    )

    testset = torch.utils.data.DataLoader(
        image_folder, batch_size=args.batch_size, shuffle=False
    )

    if args.train:
        path = args.model
        model = train_model(testset, batch_size=args.batch_size, epochs=args.epochs, no_cuda=args.no_cuda)
        torch.save(model.state_dict(), path)

    else:
        path = args.model
        model = load_model(path)
        evaluate(image_folder, model, no_cuda=args.no_cuda)


if __name__ == '__main__':
    main()
