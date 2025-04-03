# cinematic_pipeline_debugger/main.py

import os
import cv2
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
from pathlib import Path

# === CONFIG ===
FRAME_DIR = "data/frames"
DEPTH_DIR = "data/depth"
VIDEO_PATH = "data/input_clip.mp4"
MODEL_PATH = "weights/midas_v21_small-70d6b9c8.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === FRAME EXTRACTION ===
def extract_frames(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(out_dir, f"frame_{idx:04d}.png")
        cv2.imwrite(frame_path, frame)
        idx += 1
    cap.release()

# === DEPTH ESTIMATION ===
def load_midas_model(model_path):
    model = MidasNet(model_path, non_negative=True)
    model.eval()
    model.to(DEVICE)
    transform = Compose([
        Resize(384, 384),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])
    return model, transform

def predict_depth(model, transform, frame_path):
    img = cv2.imread(frame_path)[..., ::-1] / 255.0
    input_batch = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prediction = model.forward(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
        ).squeeze()
    depth = prediction.cpu().numpy()
    return depth

def save_depth_visual(depth, out_path):
    plt.imsave(out_path, depth, cmap="plasma")

# === CLI INTERFACE ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, choices=["frames", "depth"], required=True)
    parser.add_argument("--frame_idx", type=int, default=0)
    args = parser.parse_args()

    if args.step == "frames":
        extract_frames(VIDEO_PATH, FRAME_DIR)

    elif args.step == "depth":
        model, transform = load_midas_model(MODEL_PATH)
        frame_file = os.path.join(FRAME_DIR, f"frame_{args.frame_idx:04d}.png")
        depth = predict_depth(model, transform, frame_file)
        os.makedirs(DEPTH_DIR, exist_ok=True)
        depth_file = os.path.join(DEPTH_DIR, f"depth_{args.frame_idx:04d}.png")
        save_depth_visual(depth, depth_file)
        print(f"Saved depth to {depth_file}")
