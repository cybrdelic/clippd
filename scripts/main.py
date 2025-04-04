import os
import sys
import cv2
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Add MiDaS source directory to sys.path
sys.path.append("C:/Users/alexf/software-projects/clippd/midas")
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
from pathlib import Path
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2  # Direct SAM 2 model builder
from hydra import initialize, compose  # Hydra utilities for config management
from hydra.core.global_hydra import GlobalHydra  # To manage global Hydra instance

# === Configuration ===
FRAME_DIR = "../data/frames"
DEPTH_DIR = "../data/depth"
SEGMENTATION_DIR = "../data/segmentation_masks"
VIDEO_PATH = "../data/input_clip.mp4"
MODEL_PATH = "../weights/dpt_large_384.pt"
SAM_CHECKPOINT_PATH = "../weights/sam2.1_hiera_large.pt"
SAM_CONFIG_PATH = "C:/Users/alexf/software-projects/clippd/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"  # Absolute SAM 2 config file
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Frame Extraction ===
def extract_frames(video_path, out_dir):
    """Extract frames from a video and save them to the output directory."""
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

# === Depth Estimation ===
def load_midas_model(model_path):
    """Load the MiDaS model and its preprocessing transform."""
    try:
        model = DPTDepthModel(backbone="vitl16_384", non_negative=True)
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint if not isinstance(checkpoint, dict) else checkpoint.get("state_dict", checkpoint)
        print("MiDaS Weights keys and shapes (first 10):")
        for k, v in list(state_dict.items())[:10]:
            print(f"{k}: {v.shape}")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded MiDaS model with backbone 'vitl16_384' (some keys may be ignored)")
    except RuntimeError as e:
        print(f"Error loading MiDaS model weights: {e}")
        print("Run 'python run.py --model_type dpt_large_384' in MiDaS dir to verify weights")
        raise
    model.eval()
    model.to(DEVICE)
    transform = Compose([
        Resize(384, 384),
        NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        PrepareForNet()
    ])
    return model, transform

def predict_depth(model, transform, frame_paths):
    """Predict depth maps for a batch of frames."""
    samples = []
    original_shapes = []
    for frame_path in frame_paths:
        img = cv2.imread(frame_path)[..., ::-1] / 255.0  # BGR to RGB
        sample = {"image": img, "mask": np.ones(img.shape[:2], dtype=np.float32)}
        samples.append(transform(sample)["image"])
        original_shapes.append(img.shape[:2])
    input_batch = torch.from_numpy(np.stack(samples)).to(DEVICE)
    with torch.no_grad():
        predictions = model.forward(input_batch)
        depth_maps = []
        for i, prediction in enumerate(predictions):
            depth = torch.nn.functional.interpolate(
                prediction.unsqueeze(0).unsqueeze(0),
                size=original_shapes[i],
                mode="bicubic",
                align_corners=False
            ).squeeze()
            depth_maps.append(depth.cpu().numpy())
    return depth_maps

def save_depth_visual(depth, out_path):
    """Save a depth map as a visual image."""
    plt.imsave(out_path, depth, cmap="plasma")

# === Segmentation with SAM 2 ===
def load_sam_model(checkpoint_path):
    """Load the SAM 2 model using the local checkpoint and config with adjusted Hydra search path."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"SAM 2 checkpoint not found at {checkpoint_path}. Download from "
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
        )
    if not os.path.exists(SAM_CONFIG_PATH):
        raise FileNotFoundError(
            f"SAM 2 config file not found at {SAM_CONFIG_PATH}. Ensure the SAM 2 repo is correctly set up."
        )
    # Use a relative config path from the scripts/ directory
    config_dir = os.path.relpath(
        os.path.dirname(SAM_CONFIG_PATH),  # Absolute dir: C:/Users/alexf/software-projects/clippd/sam2/configs/sam2.1
        start=os.path.dirname(__file__)     # Script dir: C:/Users/alexf/software-projects/clippd/scripts
    )  # Results in "../../sam2/configs/sam2.1"
    # Clear any existing Hydra instance to avoid reinitialization errors
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path=config_dir):
        cfg = compose(config_name=os.path.basename(SAM_CONFIG_PATH))  # "sam2.1_hiera_l.yaml"
        sam_model = build_sam2(
            config_file=SAM_CONFIG_PATH,
            ckpt_path=checkpoint_path,
            device=DEVICE
        )
    predictor = SAM2ImagePredictor(sam_model)
    return predictor

def predict_segmentation(predictor, frame_path):
    """Generate a segmentation mask for a frame using SAM 2."""
    img = cv2.imread(frame_path)[..., ::-1].copy()
    predictor.set_image(img)
    masks, scores, _ = predictor.predict(
        point_coords=None,  # Automatic mask generation
        point_labels=None,
        box=None,
        multimask_output=False
    )
    return masks[0] if len(masks) > 0 else None  # Return first mask

def save_segmentation_mask(mask, out_path):
    """Save a segmentation mask as an image."""
    if mask is not None:
        plt.imsave(out_path, mask, cmap="gray")

# === Command-Line Interface ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cinematic Pipeline Debugger with Depth and Segmentation")
    parser.add_argument("--step", type=str, choices=["frames", "depth"], required=True, help="Pipeline step to execute")
    parser.add_argument("--frame_idx", type=int, default=None, help="Optional frame index for single-frame processing")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of frames per batch for depth estimation")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    if args.step == "frames":
        extract_frames(VIDEO_PATH, FRAME_DIR)
        print(f"Extracted frames to {FRAME_DIR}")

    elif args.step == "depth":
        midas_model, midas_transform = load_midas_model(MODEL_PATH)
        sam_predictor = load_sam_model(SAM_CHECKPOINT_PATH)
        os.makedirs(DEPTH_DIR, exist_ok=True)
        os.makedirs(SEGMENTATION_DIR, exist_ok=True)

        if args.frame_idx is not None:
            frame_file = os.path.join(FRAME_DIR, f"frame_{args.frame_idx:04d}.png")
            if not os.path.exists(frame_file):
                raise FileNotFoundError(f"Frame file not found: {frame_file}")
            depth = predict_depth(midas_model, midas_transform, [frame_file])[0]
            depth_file = os.path.join(DEPTH_DIR, f"depth_{args.frame_idx:04d}.png")
            save_depth_visual(depth, depth_file)
            mask = predict_segmentation(sam_predictor, frame_file)
            mask_file = os.path.join(SEGMENTATION_DIR, f"mask_{args.frame_idx:04d}.png")
            save_segmentation_mask(mask, mask_file)
            print(f"Saved depth to {depth_file} and mask to {mask_file}")
        else:
            frame_files = sorted([f for f in os.listdir(FRAME_DIR) if f.endswith(".png")])
            if not frame_files:
                raise FileNotFoundError(f"No .png files found in {FRAME_DIR}")
            batch_size = args.batch_size
            total_frames = len(frame_files)
            print(f"Processing {total_frames} frames in batches of {batch_size}...")
            for i in range(0, total_frames, batch_size):
                batch_paths = [os.path.join(FRAME_DIR, f) for f in frame_files[i:i + batch_size]]
                batch_indices = [int(f.split("_")[1].split(".")[0]) for f in frame_files[i:i + batch_size]]
                depth_maps = predict_depth(midas_model, midas_transform, batch_paths)
                for depth, idx in zip(depth_maps, batch_indices):
                    depth_file = os.path.join(DEPTH_DIR, f"depth_{idx:04d}.png")
                    mask_file = os.path.join(SEGMENTATION_DIR, f"mask_{idx:04d}.png")
                    save_depth_visual(depth, depth_file)
                    mask = predict_segmentation(sam_predictor, os.path.join(FRAME_DIR, f"frame_{idx:04d}.png"))
                    save_segmentation_mask(mask, mask_file)
                    print(f"Saved depth to {depth_file} and mask to {mask_file} ({i + 1}/{total_frames} frames processed)")
