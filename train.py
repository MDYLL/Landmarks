from model import LandmarksDataset
from model import train_transform, test_transform, Net

import torch
from torch.utils.data import DataLoader
from torch import nn

import numpy as np
from tqdm import tqdm
import pickle

import argparse


def train(train_data, test_data=None, load_dataset_from_file=True, NUM_EPOCHS=50, BATCH_SIZE=128):
    test_dataset_300W, test_dataset_Menpo = None, None
    with_test = False
    if load_dataset_from_file:
        print("Loading datasets from files ...")
        with open(train_data[0], 'rb') as f:
            train_dataset = pickle.load(f)
        if test_data:
            with_test = True
            for el in test_data:
                if "300W" in el:
                    with open(el, 'rb') as f:
                        test_dataset_300W = pickle.load(f)
                elif "Menpo" in el:
                    with open(el, 'rb') as f:
                        test_dataset_Menpo = pickle.load(f)
        print("Done")
    else:
        print("Creating datasets from source ...")
        train_dataset = LandmarksDataset(train_data, train_transform)
        if test_data:
            with_test = True
            for el in test_data:
                print(el)
                if "300W" in el:
                    test_dataset_300W = LandmarksDataset([el], test_transform)
                elif "Menpo" in el:
                    test_dataset_Menpo = LandmarksDataset([el], test_transform)
        print("Done")

    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device is {DEVICE}")

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_300W_dataloader, test_Menpo_dataloader = None, None
    if test_dataset_300W:
        test_300W_dataloader = DataLoader(dataset=test_dataset_300W, batch_size=BATCH_SIZE, shuffle=False)
    if test_dataset_Menpo:
        test_Menpo_dataloader = DataLoader(dataset=test_dataset_Menpo, batch_size=BATCH_SIZE, shuffle=False)

    model = Net().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=1e-1, factor=0.3)

    total_train_loss, total_test_300W_loss, total_test_Menpo_loss = [], [], []
    best_checkpoint_loss = 1e6
    for i in range(NUM_EPOCHS):

        train_loss = []
        model.train()
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            X_batch = batch[0].to(DEVICE)
            y_batch = batch[1].to(DEVICE)
            y_batch = y_batch / 48 - 1
            prediction = model(X_batch)
            loss = criterion(prediction, y_batch.float())
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        print(f"LR = {optimizer.param_groups[0]['lr']}")
        print(f"Epoch {i}, train MSE = {np.mean(train_loss)}")
        total_train_loss.append(np.mean(train_loss))

        mean_train_loss = np.mean(train_loss)
        scheduler.step(mean_train_loss)

        if with_test:
            model.eval()
            with torch.no_grad():
                test_300W_loss = []
                for batch in tqdm(test_300W_dataloader):
                    X_batch = batch[0].to(DEVICE)
                    y_batch = batch[1].to(DEVICE)
                    y_batch = y_batch / 48 - 1
                    prediction = model(X_batch)
                    loss = criterion(prediction, y_batch.float())
                    test_300W_loss.append(loss.item())
                print(f"Epoch {i}, 300W MSE = {np.mean(test_300W_loss)}")
                total_test_300W_loss.append(np.mean(test_300W_loss))

                test_Menpo_loss = []
                for batch in tqdm(test_Menpo_dataloader):
                    X_batch = batch[0].to(DEVICE)
                    y_batch = batch[1].to(DEVICE)
                    y_batch = y_batch / 48 - 1
                    prediction = model(X_batch)
                    loss = criterion(prediction, y_batch.float())
                    test_Menpo_loss.append(loss.item())
                print(f"Epoch {i}, Menpo MSE = {np.mean(test_Menpo_loss)}")

                total_test_Menpo_loss.append(np.mean(test_Menpo_loss))
                if total_test_Menpo_loss[-1] < best_checkpoint_loss:
                    best_checkpoint_loss = total_test_Menpo_loss[-1]
                    torch.save(model.state_dict(), "best_checkpoint.pt")
        else:
            if total_train_loss[-1] < best_checkpoint_loss:
                best_checkpoint_loss = total_train_loss[-1]
                torch.save(model.state_dict(), "best_checkpoint.pt")


def main():
    parser = argparse.ArgumentParser(description='train script', add_help=True)
    parser.add_argument("--train_data", action="store", help='', default=["train_dataset_96.pkl"], nargs='+')
    parser.add_argument("--test_data", action="store", help='',
                        default=["test_dataset_300W_96.pkl", "test_dataset_Menpo_96.pkl"], nargs='+')
    parser.add_argument("--load_dataset_from_file", action="store", type=int, help='',
                        default=1)
    parser.add_argument("--num_epochs", action="store", type=int, help='',
                        default=50)
    parser.add_argument("--batch_size", action="store", type=int, help='',
                        default=128)
    args = parser.parse_args()

    train(args.train_data, args.test_data, args.load_dataset_from_file, args.num_epochs, args.batch_size)

if __name__ == '__main__':
    main()
