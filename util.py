import math
import json
import numpy as np
import matplotlib
import cv2
from comfy.utils import ProgressBar

eps = 0.01

def draw_pose_json(pose_json, resolution_x, show_body, show_face, show_hands, pose_marker_size, face_marker_size, hand_marker_size):
    pose_imgs = []
    if pose_json:
        if pose_json.startswith('{'):
            pose_json = '[{}]'.format(pose_json)
        images = json.loads(pose_json)
        pbar = ProgressBar(len(images))
        for image in images:
            if 'people' not in image:
                pbar.update(len(images))
                return pose_imgs
            figures = image['people']
            H = image['canvas_height']
            W = image['canvas_width']
            bodies = []
            candidate = []
            subset = [[]]
            faces = []
            hands = []
            for figure in figures:
                body = []
                face = []
                lhand = []
                rhand = []
                if 'pose_keypoints_2d' in figure:
                    body = figure['pose_keypoints_2d']
                if 'face_keypoints_2d' in figure:
                    face = figure['face_keypoints_2d']
                if 'hand_left_keypoints_2d' in figure:
                    lhand = figure['hand_left_keypoints_2d']
                if 'hand_right_keypoints_2d' in figure:
                    rhand = figure['hand_right_keypoints_2d']
                if body:
                    for i in range(0,len(body),3):
                        candidate.append(body[i:i+2])
                    if not subset[0]:
                        subset[0].extend([len(subset[0])+(i//3) if body[i+2]>0 else -1 for i in range(0,len(body),3)])
                    else:
                        subset.append([len(subset[0])*len(subset)+(i//3) if body[i+2]>0 else -1 for i in range(0,len(body),3)])
                if face:
                    faces.append([face[i:i+2] for i in range(0,len(face),3)])
                if lhand:
                    hands.append([lhand[i:i+2] for i in range(0,len(lhand),3)])
                if rhand:
                    hands.append([rhand[i:i+2] for i in range(0,len(rhand),3)])

            normalized = 0.0
            if candidate:
                candidate = np.array(candidate).astype(float)
                subset = np.array(subset)
                normalized = max(np.max(candidate[...,0]),np.max(candidate[...,1]))
                if normalized>2.0:
                    candidate[...,0] /= float(W)
                    candidate[...,1] /= float(H)
            if faces:
                faces = np.array(faces).astype(float)
                normalized = max(np.max(faces[...,0]),np.max(faces[...,1]))
                if normalized>2.0:
                    faces[...,0] /= float(W)
                    faces[...,1] /= float(H)
            if hands:
                hands = np.array(hands).astype(float)
                normalized = max(np.max(hands[...,0]),np.max(hands[...,1]))
                if normalized>2.0:
                    hands[...,0] /= float(W)
                    hands[...,1] /= float(H)
            bodies = dict(candidate=candidate, subset=subset)
            pose = dict(bodies=bodies, faces=faces, hands=hands)
            pose = dict(bodies=bodies if show_body else {'candidate':[], 'subset':[]}, faces=faces if show_face else [], hands=hands if show_hands else [])
            W_scaled = resolution_x
            if resolution_x < 64:
                W_scaled = W
            H_scaled = int(H*(W_scaled*1.0/W))
            pose_img = draw_pose(pose, H_scaled, W_scaled, pose_marker_size, face_marker_size, hand_marker_size)
            pose_imgs.append(pose_img)
            pbar.update(1)

    return pose_imgs

def draw_pose(pose, H, W, pose_marker_size, face_marker_size, hand_marker_size):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if len(candidate) > 0:
        canvas = draw_bodypose(canvas, candidate, subset, pose_marker_size)

    if len(hands) > 0:
        canvas = draw_handpose(canvas, hands, hand_marker_size)

    if len(faces) > 0:
        canvas = draw_facepose(canvas, faces, face_marker_size)

    return canvas

def draw_bodypose(canvas, candidate, subset, pose_marker_size):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    # stickwidth = 4

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), pose_marker_size), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), pose_marker_size, colors[i], thickness=-1)

    return canvas


def draw_handpose(canvas, all_hand_peaks, hand_marker_size):
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for peaks in all_hand_peaks:
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=1 if hand_marker_size == 0 else hand_marker_size)

        joint_size=0
        if hand_marker_size < 2:
            joint_size = hand_marker_size + 1
        else:
            joint_size = hand_marker_size + 2
        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), joint_size, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas, all_lmks, face_marker_size):
    H, W, C = canvas.shape
    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), face_marker_size, (255, 255, 255), thickness=-1)
    return canvas
