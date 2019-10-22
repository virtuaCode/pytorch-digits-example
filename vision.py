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

BATCH_SIZE = 16
EPOCHS = 4

preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(28),
    transforms.CenterCrop(28),
    transforms.ToTensor(),
])


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = fn.relu(self.fc1(x))
        x = fn.relu(self.fc2(x))
        x = fn.relu(self.fc3(x))
        x = self.fc4(x)

        return fn.log_softmax(x, dim=1)


def train_model(testset, no_cuda=False):
    training = datasets.MNIST("", train=True, download=True,
                              transform=transforms.Compose([transforms.ToTensor()]))

    trainset = torch.utils.data.DataLoader(
        training, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device('cpu' if no_cuda else 'cuda')

    net = Net().to(device)
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        print("Begin Epoch", epoch + 1, "/", EPOCHS)
        for (data, target) in tqdm(trainset):
            data, target = data.to(device), target.to(device)
            net.zero_grad()
            output = net(data.view(-1, 28*28))
            loss = fn.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        print('Loss:', loss.item())
        with torch.no_grad():
            count = 0
            count_matches = 0
            for (data, target) in testset:
                data = data.to(device)
                output = net(data.view(-1, 28*28))
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


def evaluate(imageset, model, no_cuda=False):
    device = torch.device('cpu' if no_cuda else 'cuda')
    model.to(device)
    model.eval()
    with torch.no_grad():
        size = len(imageset)
        for idx, (data, target) in enumerate(imageset):
            plt.subplot(1, size, idx + 1)
            plt.axis('off')
            plt.imshow(data.view(28, 28))
            data = data.to(device)
            output = model(data.view(-1, 28*28))
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
    parser.add_argument('--no-cuda', action='store_true', default=False)                        

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
        image_folder, batch_size=BATCH_SIZE, shuffle=False
    )

    #print(image_folder.class_to_idx)
    #exit()

    if args.train:
        model = train_model(testset, no_cuda=args.no_cuda)
        path = args.model
        torch.save(model.state_dict(), path)

    else:
        path = args.model
        model = load_model(path)
        evaluate(image_folder, model, no_cuda=args.no_cuda)


if __name__ == '__main__':
    main()
