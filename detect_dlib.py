from tqdm import tqdm
import glob
from utils import get_data
import dlib
import os
import numpy as np
import argparse

def detect(input_path, output_path, normalization_path):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    file_norm = open(normalization_path, 'w')
    os.makedirs(output_path, exist_ok=True)
    for extension in ["*jpg", "*png"]:
        for f in tqdm(glob.glob(os.path.join(input_path, extension))):
            data = get_data(f)
            if data is None:
                continue
            img = dlib.load_rgb_image(f)
            key_points, left, right, top, bottom, height, width = data
            bbox = dlib.rectangle(left, top, right, bottom)
            dlib_pred = predictor(img, bbox)
            dlib_keypoints = np.zeros((68, 2))
            for i in range(68):
                dlib_keypoints[i][0] = dlib_pred.part(i).x
                dlib_keypoints[i][1] = dlib_pred.part(i).y
            with open(os.path.join(output_path, f.split('/')[-1][:-3] + "pts"), 'w') as fout:
                for i in range(68):
                    fout.write(str(dlib_keypoints[i][0]) + ' ' + str(dlib_keypoints[i][1]) + '\n')

            file_norm.write(f.split('/')[-1][:-3] + "pts" + ' ' + str(np.sqrt((right - left) * (bottom - top))) + '\n')

def main():
    parser = argparse.ArgumentParser(description='detect script', add_help=True)
    parser.add_argument("--input_path", action="store", type=str, help='', default="landmarks_task/Menpo/test")
    parser.add_argument("--output_path", action="store", type=str, help='',
                        default="landmarks_task/Menpo/dlib")
    parser.add_argument("--normalization_path", action="store", type=str, help='',
                        default="landmarks_task/Menpo/normalization.pts")

    args = parser.parse_args()

    detect(args.input_path, args.output_path, args.normalization_path)


if __name__ == '__main__':
    main()