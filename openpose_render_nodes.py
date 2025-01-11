import json
import numpy as np
import torch
from .util import draw_pose, draw_pose_json

class OpenposeRenderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "show_body": ("BOOLEAN", {"default": True}),
                "show_face": ("BOOLEAN", {"default": True}),
                "show_hands": ("BOOLEAN", {"default": True}),
                "resolution_x": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 12800
                }),
                "pose_marker_size": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 100
                }),
                "face_marker_size": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 100
                }),
                "hand_marker_size": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 100
                }),
                "POSE_JSON": ("STRING", {"multiline": True}),
                "POSE_KEYPOINT": ("POSE_KEYPOINT",{"default": None}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render_img"
    CATEGORY = "ultimate-openpose"

    def render_img(self, show_body, show_face, show_hands, resolution_x, pose_marker_size, face_marker_size, hand_marker_size, POSE_JSON, POSE_KEYPOINT=None):
        if POSE_KEYPOINT is not None:
            POSE_JSON = json.dumps(POSE_KEYPOINT).replace("'",'"').replace('None','[]')
        elif POSE_JSON:
            POSE_JSON = POSE_JSON.replace("'",'"').replace('None','[]')

        pose_imgs = draw_pose_json(POSE_JSON, resolution_x, show_body, show_face, show_hands, pose_marker_size, face_marker_size, hand_marker_size)
        if pose_imgs:
            pose_imgs_np = np.array(pose_imgs).astype(np.float32) / 255
            return (torch.from_numpy(pose_imgs_np),)
        else:
            raise ValueError("Invalid input type. Expected an input to give an output.")
