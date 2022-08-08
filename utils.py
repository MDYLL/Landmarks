import numpy as np
import dlib


def read_pts(filename):
    landmarks = []
    with open(filename) as f:
        for line in f:
            if line.startswith("version"):
                continue
            if line.startswith("n_points"):
                if line.split()[-1] != "68":
                    return None
                else:
                    continue
            point = line.split()
            if len(point) != 2:
                continue
            landmarks.append([float(el) for el in point])
    return np.array(landmarks)


def get_data(image_filename):
    detector = dlib.get_frontal_face_detector()
    key_points = read_pts(image_filename[:-3] + "pts")
    if key_points is None:
        return None

    img = dlib.load_rgb_image(image_filename)
    height, width, _ = img.shape

    for upsample in range(1):  # If you have 2 hours change range(1) -> [1,0]
        dets = detector(img, upsample)
        if len(dets) == 0:
            return None

        for det in dets:
            if det.top() <= key_points[30, 1] <= det.bottom() and \
                    det.left() <= key_points[30, 0] <= det.right():
                left = max(0, det.left())
                top = max(0, det.top())
                right = min(img.shape[1], det.right())
                bottom = min(img.shape[0], det.bottom())
                left = int(left)
                top = int(top)
                right = int(right)
                bottom = int(bottom)

                return key_points, left, right, top, bottom, height, width
    return None
