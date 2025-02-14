import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data import CIFAR10Dataset
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data_utils
import torch.nn.functional as F
from Models import *
from train_utils import *
import torch.utils.data as utils
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01,
                        help="initial_learning_rate")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--num_classes", type=int,
                        default=10, help="number of classes")
    parser.add_argument(
        "--h", type=int, default=2, help="dimension of the hidden layer"
    )
    parser.add_argument(
        "--scale", type=float, default=2, help="scaling factor for distance"
    )
    parser.add_argument(
        "--reg", type=float, default=0.001, help="regularization coefficient"
    )
    parser.add_argument(
        "--exp_name", type=str, default="default", help="name for saving results"
    )

    args, _ = parser.parse_known_args()

    def reshape_dataset(dataset, height, width):
        new_dataset = []
        for k in range(0, dataset.shape[0]):
            new_dataset.append(np.reshape(dataset[k], [1, height, width]))

        return np.array(new_dataset)

    
    dataset = CIFAR10Dataset(batch_size = args.batch_size)
    trainloader = dataset.trainloader
    testloader = dataset.testloader
    classes = dataset.classes

    model = ModifiedResNet(args.h, args.num_classes, args.scale).to(device)

    lrate = args.lr
    optimizer_s = optim.SGD(
        model.parameters(), lr=lrate, momentum=0.9, weight_decay=1e-4
    )

    num_epochs = 100

    # Filename to save plots. Three plots are updated with each epoch; Accuracy, Loss and Error Rate
    plotsFileName = "cifar10"
    # Filename to save training log. Updated with each epoch, contains Accuracy, Loss and Error Rate

    train_model(
        model,
        optimizer_s,
        lrate,
        num_epochs,
        args.reg,
        trainloader,
        testloader,
        len(trainloader.dataset),
        len(testloader.dataset),
        plotsFileName,
        args.exp_name,
        device
    )