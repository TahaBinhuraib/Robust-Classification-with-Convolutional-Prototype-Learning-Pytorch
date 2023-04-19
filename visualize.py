import argparse

from Models import *

from data import CIFAR10Dataset
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import os
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--save", type=str, help="Path to save")
    parser.add_argument("--s", type=float, default=2, help="Scale factor")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units")



    args = parser.parse_args()

    dataset = CIFAR10Dataset(batch_size = 64)
    trainloader = dataset.trainloader
    testloader = dataset.testloader
    classes = dataset.classes

    model = ModifiedResNet(args.hidden_units, 10, args.s).to(device)
    
    model = torch.load(args.checkpoint) # bad programming example 
    model.eval()

    train_features = []
    train_labels = []

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        features, _, _, _ = model(inputs)
        train_features.append(features.cpu().detach().numpy())
        train_labels.append(labels.cpu().detach().numpy())

    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    test_features = []
    test_labels = []

    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        features, _, _, _ = model(inputs)
        test_features.append(features.cpu().detach().numpy())
        test_labels.append(labels.cpu().detach().numpy())

    test_features = np.concatenate(test_features, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)



    centers = model.dce_loss.centers.cpu().detach().numpy()
    centers = centers.transpose(1, 0)

    # train_features = np.concatenate([train_features, centers], axis=0)
    # train_features = TSNE(n_components=2).fit_transform(train_features)
    # centers = train_features[-10:, :]
    # train_features = train_features[:-10, :]


    pca = PCA(n_components=2)

    train_f_number = train_features.shape[0]

    
    all_parameters = np.concatenate([train_features,test_features,centers],axis=0) # concacatanate all parameters 
    all_parameters = pca.fit_transform(all_parameters) # projection 
    
    train_features = all_parameters[:train_f_number,:]
    test_features = all_parameters[train_f_number:-10,:]
    centers = all_parameters[-10:,:]

    
    plt.figure(figsize=(10, 10))
    plt.scatter(train_features[:, 0], train_features[:, 1], c=train_labels, cmap='tab10')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100)
    plt.savefig(os.path.join(args.save, "train_pca.png"))

    # centers = model.dce_loss.centers.cpu().detach().numpy()
    # centers = centers.transpose(1, 0)
    # test_features = np.concatenate([test_features, centers], axis=0)
    # test_features = TSNE(n_components=2).fit_transform(test_features)
    # centers = test_features[-10:, :]
    # test_features = test_features[:-10, :]

    plt.figure(figsize=(10, 10))
    plt.scatter(test_features[:, 0], test_features[:, 1], c=test_labels, cmap='tab10')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100)
    plt.savefig(os.path.join(args.save, "test_pca.png"))


    plt.close('all')

    print("Done")


