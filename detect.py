import os
import torch
from tqdm import tqdm
import argparse
import pickle
from model import Net, LandmarksDataset


def detect(model_path, dataset_pickle_path, output_path):
    NEW_SIZE = (96, 96)
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device is {DEVICE}")
    with open(dataset_pickle_path, 'rb') as f:
        dataset = pickle.load(f)
    model = Net()
    if DEVICE == torch.device('cpu'):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    os.makedirs(output_path, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataset):
            image, key_points, img_path, left, top, right, bottom = data
            pred = model(image[None, ...].to(DEVICE))
            pred = pred.cpu().detach().numpy()
            pred = (pred + 1) * 48
            pred = pred.reshape(68, 2)
            pred[:, 0] = pred[:, 0] * (right - left) / NEW_SIZE[0] + left
            pred[:, 1] = pred[:, 1] * (bottom - top) / NEW_SIZE[0] + top
            with open(os.path.join(output_path, img_path.split('/')[-1][:-3] + "pts"), 'w') as fout:
                for i in range(68):
                    fout.write(str(pred[i][0]) + ' ' + str(pred[i][1]) + '\n')


def main():
    parser = argparse.ArgumentParser(description='detect script', add_help=True)
    parser.add_argument("--model_path", action="store", type=str, help='', default="best_checkpoint_0708.pt")
    parser.add_argument("--dataset_pickle_path", action="store", type=str, help='',
                        default="test_dataset_Menpo_96.pkl")
    parser.add_argument("--output_path", action="store", type=str, help='',
                        default="landmarks_task/Menpo/results")

    args = parser.parse_args()

    detect(args.model_path, args.dataset_pickle_path, args.output_path)


if __name__ == '__main__':
    main()
