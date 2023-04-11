import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm
import scipy.misc
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim
import torchvision
import numpy as np
import torch.utils.data as data_utils
import torch.nn.functional as F
from Models import *
import os 


def lr_scheduler(optimizer, init_lr, epoch):

    for param_group in optimizer.param_groups:

        if epoch == 20 or epoch == 25:
            param_group["lr"] = param_group["lr"] / 10

        if epoch == 0:
            param_group["lr"] = init_lr

        print("Current learning rate is {}".format(param_group["lr"]))

    return optimizer


def train_model(
    cnn,
    optimizer_s,
    lrate,
    num_epochs,
    reg,
    train_loader,
    test_loader,
    dataset_train_len,
    dataset_test_len,
    plotsFileName,
    save_folder,
    device
):  
    
    if not os.path.exists(f"results/{save_folder}"):
        os.makedirs(f"results/{save_folder}")
    epochs = []
    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []
    train_error = []
    test_error = []
    best_acc = 0.0
    for epoch in range(num_epochs):
        cnn.train()
        epochs.append(epoch)
        optimizer = lr_scheduler(optimizer_s, lrate, epoch)
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("*" * 70)
        running_loss = 0.0
        running_corrects = 0.0
        train_batch_ctr = 0.0

        for i, (image, label) in enumerate(tqdm(train_loader)):

            image, label = Variable(image.to(device), requires_grad=True), Variable(
                label.to(device), requires_grad=False
            )

            optimizer.zero_grad()
            features, centers, distance, outputs = cnn(image)

            _, preds = torch.max(distance, 1)

            loss1 = F.nll_loss(outputs, label)
            loss2 = regularization(features, centers, label)

            loss = loss1 + reg * loss2

            loss.backward()

            optimizer.step()
            train_batch_ctr = train_batch_ctr + 1

            running_loss += loss.item()

            running_corrects += torch.sum(preds == label.data)

            epoch_acc = float(running_corrects) / (float(dataset_train_len))

        print(
            "Train corrects: {} Train samples: {} Train accuracy: {}".format(
                running_corrects, (dataset_train_len), epoch_acc
            )
        )
        train_acc.append(epoch_acc)
        train_loss.append(1.0 * running_loss / train_batch_ctr)
        train_error.append(
            ((dataset_train_len) - running_corrects) / (dataset_train_len)
        )

        cnn.eval()
        test_running_corrects = 0.0
        test_batch_ctr = 0.0
        test_running_loss = 0.0
        test_total = 0.0

        for image, label in test_loader:

            with torch.no_grad():
                image, label = Variable(
                    image.to(device)), Variable(label.to(device))

                features, centers, distance, test_outputs = cnn(image)
                _, predicted_test = torch.max(distance, 1)

                loss1 = F.nll_loss(test_outputs, label)
                loss2 = regularization(features, centers, label)

                loss = loss1 + reg * loss2

                test_running_loss += loss.item()
                test_batch_ctr = test_batch_ctr + 1

                test_running_corrects += torch.sum(
                    predicted_test == label.data)
                test_epoch_acc = float(
                    test_running_corrects) / float(dataset_test_len)

        if test_epoch_acc > best_acc:
            torch.save(cnn, f"results/{save_folder}/best_model.pt")
            best_acc = test_epoch_acc

        test_acc.append(test_epoch_acc)
        test_loss.append(1.0 * test_running_loss / test_batch_ctr)

        print(
            "Test corrects: {} Test samples: {} Test accuracy {}".format(
                test_running_corrects, (dataset_test_len), test_epoch_acc
            )
        )

        print(
            "Train loss: {} Test loss: {}".format(
                train_loss[epoch], test_loss[epoch])
        )

        print("*" * 70)

        plots(epochs, train_acc, test_acc, train_loss, test_loss,save_folder, plotsFileName)

    torch.save(cnn, f"results/{save_folder}/final_model.pt")


def plots(epochs, train_acc, test_acc, train_loss, test_loss, folder_name, plotsFileName):
    """Helper function to plot the training and test loss and accuracy; it saved it but dont show it"""
    # Fig size should be changed
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    ax1.plot(epochs, train_acc, label='Train')
    ax1.plot(epochs, test_acc, label='Test')
    ax1.set_title('accuracy')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.legend()

    ax2.plot(epochs, train_loss, label='Train')
    ax2.plot(epochs, test_loss, label='Test')
    ax2.set_title('loss')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.legend()

    # Save the plot
    plt.savefig("results/" + folder_name + "/" + plotsFileName)
