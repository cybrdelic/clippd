#!/usr/bin/env python3
"""
An **ultra-detailed**, intricately engineered pipeline for 3D scene understanding,
mesh generation, and STL export—built for the visionary who wants it all.
This pipeline leverages MiDaS for dense relative depth estimation, enhances
depth maps with robust fusion (RANSAC-based scale-shift correction, multi-layer segmentation),
and then converts those depth maps into fully meshed 3D models with STL export.
It also integrates instance segmentation for object-level scene understanding.
Enhanced for perfect UX: no hangs, exhaustive logging, profiling, and user control.
"""

import os
import sys
import cv2
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision
from scipy import ndimage
from skimage import segmentation, morphology, feature
from skimage.measure import marching_cubes
from pathlib import Path
import math
import time
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as ThreadTimeoutError
import cProfile
import pstats

# Added for robust regression (RANSAC)
from sklearn.linear_model import RANSACRegressor

# For STL generation
try:
    from stl import mesh as stl_mesh
    STL_AVAILABLE = True
except ImportError:
    STL_AVAILABLE = False
    print("[WARN] numpy-stl not found. STL export will use a basic ASCII writer.")

# Add MiDaS source directory to sys.path
sys.path.append("C:/Users/alexf/software-projects/clippd/midas")
try:
    from midas.dpt_depth import DPTDepthModel
    from midas.transforms import Resize, NormalizeImage, PrepareForNet
except ImportError as e:
    print(f"[ERROR] Failed to import MiDaS modules: {e}")
    sys.exit(1)

from torchvision.transforms import Compose

# SAM 2 imports
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.build_sam import build_sam2
    from hydra import initialize, compose
    from hydra.core.global_hydra import GlobalHydra
except ImportError as e:
    print(f"[ERROR] Failed to import SAM 2 modules: {e}")
    sys.exit(1)

# For 3D plotting (if needed)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# === Configuration ===
FRAME_DIR = "../data/frames"
DEPTH_DIR = "../data/depth"
SEGMENTATION_DIR = "../data/segmentation_masks"
LAYERED_SEG_DIR = "../data/layered_segmentation"
FUSION_DIR = "../data/depth_fusion"
MESH_DIR = "../data/mesh"
SCENE_DIR = "../data/scene"
VIDEO_PATH = "../data/input_clip.mp4"
MODEL_PATH = "../weights/dpt_large_384.pt"
SAM_CHECKPOINT_PATH = "../weights/sam2.1_hiera_large.pt"
SAM_CONFIG_PATH = "C:/Users/alexf/software-projects/clippd/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_DEPTH_LAYERS = 7
DEFAULT_FOCAL = 1.2e3
TIMEOUT_SECONDS = 30  # Tightened to 30 seconds for aggressive hang detection

# === Utility Functions ===
def ensure_dir(directory):
    """Ensure a directory exists, with feedback."""
    try:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory ensured: {directory}")
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {e}")
        sys.exit(1)

def timeout_handler(func, *args, timeout=TIMEOUT_SECONDS, **kwargs):
    """Execute a function with a timeout to prevent hanging."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except ThreadTimeoutError:
            logger.error(f"Function {func.__name__} timed out after {timeout} seconds")
            raise TimeoutError(f"Operation {func.__name__} exceeded {timeout} seconds")

def profile_function(func, *args, **kwargs):
    """Profile a function and log its performance stats."""
    profiler = cProfile.Profile()
    result = profiler.runcall(func, *args, **kwargs)
    ps = pstats.Stats(profiler).sort_stats('cumulative')
    logger.debug("Profiling stats:")
    ps.print_stats(10)  # Print top 10 time-consuming calls
    return result

# === Frame Extraction ===
def extract_frames(video_path, out_dir):
    """Extract frames from a video with progress updates."""
    ensure_dir(out_dir)
    logger.info(f"Starting frame extraction from {video_path}")
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {video_path}")
        sys.exit(1)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Total frames to extract: {frame_count}")
    idx = 0
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(out_dir, f"frame_{idx:04d}.png")
        cv2.imwrite(out_path, frame)
        idx += 1
        if idx % 100 == 0 or idx == frame_count:
            elapsed = time.time() - start_time
            logger.info(f"Extracted {idx}/{frame_count} frames ({(idx/frame_count)*100:.1f}%) in {elapsed:.2f}s")
    cap.release()
    logger.info(f"Frame extraction completed: {idx} frames saved to {out_dir}")

# === Depth Estimation ===
def load_midas_model(model_path):
    """Load the MiDaS model and its preprocessing transform."""
    logger.info(f"Loading MiDaS model from {model_path}")
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = DPTDepthModel(backbone="vitl16_384", non_negative=True)
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint if not isinstance(checkpoint, dict) else checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(DEVICE)
        transform = Compose([
            Resize(384, 384),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet()
        ])
        logger.info("MiDaS model loaded successfully")
        return model, transform
    except Exception as e:
        logger.error(f"Failed to load MiDaS model: {e}")
        sys.exit(1)

def predict_depth(model, transform, frame_paths):
    """Predict depth maps for a batch of frames with enhanced detail."""
    logger.info(f"Predicting depth for {len(frame_paths)} frames")
    samples = []
    original_shapes = []
    for frame_path in frame_paths:
        if not os.path.exists(frame_path):
            logger.error(f"Frame not found: {frame_path}")
            continue
        logger.debug(f"Loading frame: {frame_path}")
        img = cv2.imread(frame_path)
        if img is None:
            logger.error(f"Failed to load image: {frame_path}")
            continue
        img = img[..., ::-1] / 255.0
        sample = {"image": img, "mask": np.ones(img.shape[:2], dtype=np.float32)}
        samples.append(transform(sample)["image"])
        original_shapes.append(img.shape[:2])
    if not samples:
        logger.error("No valid frames to process for depth estimation")
        return []
    input_batch = torch.from_numpy(np.stack(samples)).to(DEVICE)
    logger.info(f"Processing depth batch of size {len(samples)} on {DEVICE}")
    start_time = time.time()
    with torch.no_grad():
        predictions = timeout_handler(model.forward, input_batch, timeout=60)
    depth_maps = []
    for i, prediction in enumerate(predictions):
        logger.debug(f"Interpolating depth map {i+1}/{len(predictions)}")
        depth = torch.nn.functional.interpolate(
            prediction.unsqueeze(0).unsqueeze(0),
            size=original_shapes[i],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()
        depth_maps.append(depth)
    logger.info(f"Depth prediction completed in {time.time() - start_time:.2f}s")
    return depth_maps

def save_depth_visual(depth, out_path):
    """Save a depth map as a visual image using the plasma colormap."""
    ensure_dir(os.path.dirname(out_path))
    logger.info(f"Saving depth map to {out_path}")
    plt.imsave(os.path.normpath(out_path), depth, cmap="plasma")
    logger.info(f"Depth map saved successfully")

# === Instance Segmentation Model ===
def load_instance_segmentation_model():
    """Load a pre-trained instance segmentation model (Mask R-CNN)."""
    logger.info("Loading Mask R-CNN model")
    try:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        model.to(DEVICE)
        logger.info("Mask R-CNN loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load Mask R-CNN: {e}")
        sys.exit(1)

def load_sam_model(checkpoint_path):
    """Load the SAM 2 model using the local checkpoint and config."""
    logger.info(f"Loading SAM 2 model from {checkpoint_path}")
    if not os.path.exists(checkpoint_path) or not os.path.exists(SAM_CONFIG_PATH):
        logger.error(f"SAM 2 files not found: checkpoint={checkpoint_path}, config={SAM_CONFIG_PATH}")
        sys.exit(1)
    config_dir = os.path.relpath(os.path.dirname(SAM_CONFIG_PATH), start=os.path.dirname(__file__))
    GlobalHydra.instance().clear()
    try:
        with initialize(version_base=None, config_path=config_dir):
            cfg = compose(config_name=os.path.basename(SAM_CONFIG_PATH))
            sam_model = build_sam2(
                config_file=SAM_CONFIG_PATH,
                ckpt_path=checkpoint_path,
                device=DEVICE
            )
        logger.info("SAM 2 model loaded successfully")
        return SAM2ImagePredictor(sam_model)
    except Exception as e:
        logger.error(f"Failed to load SAM 2 model: {e}")
        sys.exit(1)

# === Depth-based Segmentation ===
def segment_by_depth_layers(depth_map, num_layers=NUM_DEPTH_LAYERS):
    """Segment the image into multiple depth layers for enhanced scene intricacy."""
    logger.info(f"Segmenting depth map into {num_layers} layers")
    start_time = time.time()
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    depth_layers = np.zeros_like(depth_norm, dtype=np.uint8)
    layer_masks = []
    thresholds = np.linspace(0, 1, num_layers + 1)
    for i in range(num_layers):
        logger.debug(f"Processing layer {i+1}/{num_layers}")
        lower, upper = thresholds[i], thresholds[i + 1]
        layer_mask = ((depth_norm >= lower) & (depth_norm < upper)).astype(np.uint8)
        if np.sum(layer_mask) > 0:
            layer_mask = morphology.binary_closing(layer_mask, morphology.disk(5))
            layer_mask = ndimage.binary_fill_holes(layer_mask).astype(np.uint8)
            layer_mask = morphology.remove_small_objects(layer_mask.astype(bool), min_size=300).astype(np.uint8)
        layer_masks.append(layer_mask)
        depth_layers[layer_mask > 0] = i + 1
    logger.info(f"Depth segmentation completed in {time.time() - start_time:.2f}s")
    return layer_masks, depth_layers

# === Visualization Overlay ===
def overlay_depth_reliability(depth_map, reliability_map):
    """Overlay the reliability map on the depth map for visual feedback."""
    logger.info("Generating depth reliability overlay")
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    reliability_norm = cv2.normalize(reliability_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
    reliability_vis = cv2.applyColorMap(reliability_norm, cv2.COLORMAP_HOT)
    overlay = cv2.addWeighted(depth_vis, 0.7, reliability_vis, 0.3, 0)
    logger.info("Overlay generation completed")
    return overlay

# === DeepRelativeFusion Implementation ===
def process_frame_with_deeprelativefusion(frame_path, depth_map, midas_model, midas_transform):
    """Enhanced DeepRelativeFusion with detailed logging."""
    logger.info(f"Processing frame with DeepRelativeFusion: {frame_path}")
    start_time = time.time()
    if not os.path.exists(frame_path):
        logger.error(f"Frame file not found: {frame_path}")
        sys.exit(1)
    frame = cv2.imread(frame_path)
    if frame is None:
        logger.error(f"Failed to load frame: {frame_path}")
        sys.exit(1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    logger.debug("Extracting high-gradient regions")
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    gradient_threshold = np.percentile(gradient_magnitude, 70)
    high_gradient_mask = gradient_magnitude > gradient_threshold

    semi_dense_depth = np.zeros_like(depth_map)
    semi_dense_depth[high_gradient_mask] = depth_map[high_gradient_mask]

    reliability_map = np.zeros_like(depth_map)
    normalized_gradient = gradient_magnitude / (np.max(gradient_magnitude) + 1e-8)
    reliability_map[high_gradient_mask] = normalized_gradient[high_gradient_mask]

    logger.debug("Performing RANSAC scale-shift correction")
    valid_mask = high_gradient_mask & (semi_dense_depth > 0)
    if np.sum(valid_mask) > 100:
        y_vals = semi_dense_depth[valid_mask].flatten()
        x_vals = depth_map[valid_mask].flatten().reshape(-1, 1)
        ransac = RANSACRegressor(min_samples=0.5, residual_threshold=0.1)
        ransac.fit(x_vals, y_vals)
        scale = ransac.estimator_.coef_[0]
        shift = ransac.estimator_.intercept_
        logger.debug(f"RANSAC scale: {scale:.4f}, shift: {shift:.4f}")
        corrected_depth = scale * depth_map + shift
    else:
        logger.warning("Insufficient high-gradient points for RANSAC; using raw depth")
        corrected_depth = depth_map.copy()

    logger.debug("Fusing depth maps")
    alpha = 0.7
    fused_depth = corrected_depth.copy()
    valid_semi = semi_dense_depth > 0
    fused_depth[valid_semi] = alpha * semi_dense_depth[valid_semi] + (1 - alpha) * corrected_depth[valid_semi]

    logger.debug("Applying filters")
    try:
        fused_depth = cv2.ximgproc.guidedFilter(guide=frame, src=fused_depth.astype(np.float32), radius=9, eps=0.01)
    except AttributeError:
        logger.warning("cv2.ximgproc not available; skipping guided filter")
    fused_depth = cv2.medianBlur(fused_depth.astype(np.float32), 5)
    fused_depth = cv2.bilateralFilter(fused_depth.astype(np.float32), d=9, sigmaColor=0.05, sigmaSpace=5)

    logger.debug("Generating visualization")
    fusion_vis = np.zeros((frame.shape[0], frame.shape[1] * 4, 3), dtype=np.uint8)
    def prepare_vis(map_data):
        norm = cv2.normalize(map_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_PLASMA)
    semi_dense_vis = prepare_vis(semi_dense_depth)
    corrected_vis = prepare_vis(corrected_depth)
    fused_vis = prepare_vis(fused_depth)
    overlay_vis = overlay_depth_reliability(fused_depth, reliability_map)

    fusion_vis[:, 0:frame.shape[1]] = semi_dense_vis
    fusion_vis[:, frame.shape[1]:frame.shape[1]*2] = corrected_vis
    fusion_vis[:, frame.shape[1]*2:frame.shape[1]*3] = fused_vis
    fusion_vis[:, frame.shape[1]*3:] = overlay_vis

    labels = ["Semi-Dense Depth", "Corrected MiDaS", "Fused Depth", "Overlay (Fused+Reliability)"]
    for i, label in enumerate(labels):
        cv2.putText(fusion_vis, label, (frame.shape[1] * i + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    vis_dir = os.path.join(os.path.dirname(frame_path), "..", "fusion_vis")
    ensure_dir(vis_dir)
    vis_path = os.path.join(vis_dir, f"fusion_{os.path.basename(frame_path)}")
    cv2.imwrite(vis_path, fusion_vis)
    logger.info(f"Fusion visualization saved to {vis_path} in {time.time() - start_time:.2f}s")

    return fused_depth, semi_dense_depth, reliability_map

# === Vanishing Point SLAM with Enhanced Depth Correction ===
def process_frame_with_vp_slam(frame_path, depth_map, hough_threshold=150, max_iterations=5000):
    """
    Enhance the depth map by incorporating vanishing point analysis.
    Optimized to prevent hangs with exhaustive logging, profiling, tight timeouts, and user control.
    """
    logger.info(f"Starting VP-SLAM processing for frame: {frame_path} (Hough threshold: {hough_threshold}, max iterations: {max_iterations})")
    start_time = time.time()

    # Check frame existence and load
    if not os.path.exists(frame_path):
        logger.error(f"Frame file not found: {frame_path}")
        raise FileNotFoundError(f"Frame file not found: {frame_path}")
    frame = timeout_handler(cv2.imread, frame_path, timeout=5)
    if frame is None:
        logger.error(f"Failed to load frame: {frame_path}")
        raise ValueError(f"Failed to load frame: {frame_path}")
    logger.debug(f"Frame loaded: {frame.shape}")

    # Edge detection
    logger.debug("Performing edge detection")
    gray = timeout_handler(cv2.cvtColor, frame, cv2.COLOR_BGR2GRAY, timeout=5)
    edges = timeout_handler(cv2.Canny, gray, 50, 150, apertureSize=3, timeout=10)
    logger.debug(f"Edges detected: {edges.shape}")

    # Hough Line Transform with tight timeout
    logger.debug(f"Detecting lines with Hough Transform (threshold={hough_threshold})")
    try:
        lines = timeout_handler(cv2.HoughLines, edges, 1, np.pi/180, threshold=hough_threshold, timeout=15)
    except TimeoutError:
        logger.warning("HoughLines timed out after 15 seconds; falling back to original depth")
        return depth_map, frame.copy()
    if lines is None or len(lines) == 0:
        logger.warning("No lines detected for VP-SLAM; falling back to original depth")
        return depth_map, frame.copy()
    logger.debug(f"Detected {len(lines)} lines")

    # Visualize lines (limited to first 100 to avoid overload)
    vis_image = frame.copy()
    for i, line in enumerate(lines[:100]):  # Cap at 100 lines for visualization
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + 2000 * (-b)), int(y0 + 2000 * (a))
        x2, y2 = int(x0 - 2000 * (-b)), int(y0 - 2000 * (a))
        cv2.line(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    logger.debug("Lines visualized (capped at 100)")

    # Group lines by orientation with timeout
    logger.debug("Grouping lines by orientation")
    groups = {}
    try:
        for line in timeout_handler(lambda x: x, lines, timeout=5):  # Simple passthrough with timeout
            rho, theta = line[0]
            if theta > np.pi/2:
                theta -= np.pi
                rho = -rho
            bin_idx = int(theta * 18 / np.pi)
            groups.setdefault(bin_idx, []).append((rho, theta))
        dominant_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)[:3]
        logger.debug(f"Found {len(dominant_groups)} dominant line groups")
    except TimeoutError:
        logger.warning("Line grouping timed out; falling back to original depth")
        return depth_map, vis_image

    # Find vanishing points with iteration limit and tight timeouts
    logger.debug("Computing vanishing points")
    vanishing_points = []
    iteration_count = 0
    for i, (bin_i, lines_i) in enumerate(dominant_groups):
        for j, (bin_j, lines_j) in enumerate(dominant_groups[i+1:], start=i+1):
            for rho_i, theta_i in lines_i:
                for rho_j, theta_j in lines_j:
                    if iteration_count >= max_iterations:
                        logger.warning(f"Reached max iterations ({max_iterations}) in vanishing point calculation; proceeding with {len(vanishing_points)} points")
                        break
                    iteration_count += 1
                    A = np.array([[np.cos(theta_i), np.sin(theta_i)],
                                  [np.cos(theta_j), np.sin(theta_j)]])
                    b_vec = np.array([rho_i, rho_j])
                    try:
                        x, y = timeout_handler(np.linalg.solve, A, b_vec, timeout=0.5)
                        h, w = frame.shape[:2]
                        margin = 2 * max(h, w)
                        if -margin <= x <= w + margin and -margin <= y <= h + margin:
                            vanishing_points.append((x, y))
                    except (np.linalg.LinAlgError, TimeoutError):
                        continue
            if iteration_count >= max_iterations:
                break
        if iteration_count >= max_iterations:
            break
    logger.debug(f"Found {len(vanishing_points)} vanishing points after {iteration_count} iterations")

    # Apply depth correction if vanishing points exist
    if vanishing_points:
        logger.debug("Applying depth correction based on vanishing points")
        for vp in vanishing_points[:10]:  # Cap at 10 VPs to avoid overload
            x, y = vp
            h, w = frame.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(vis_image, (int(x), int(y)), 10, (0, 255, 0), -1)
        enhanced_depth = depth_map.copy()
        h, w = frame.shape[:2]
        principal_point = (w/2, h/2)
        correction_factors = []
        for vp in vanishing_points[:10]:  # Cap at 10 for performance
            x, y = vp
            dx = x - principal_point[0]
            dy = y - principal_point[1]
            distance = np.sqrt(dx*dx + dy*dy)
            if distance < 10:
                continue
            dx /= distance
            dy /= distance
            try:
                y_coords, x_coords = timeout_handler(np.mgrid.__getitem__, (slice(0, h), slice(0, w)), timeout=5)
                pixel_dx = x_coords - principal_point[0]
                pixel_dy = y_coords - principal_point[1]
                pixel_distances = timeout_handler(np.sqrt, pixel_dx*pixel_dx + pixel_dy*pixel_dy, timeout=5) + 1e-8
                pixel_dx /= pixel_distances
                pixel_dy /= pixel_distances
                alignment = pixel_dx * dx + pixel_dy * dy
                correction_factor = 0.3 + 0.7 * np.abs(alignment)
                correction_factors.append(correction_factor)
            except TimeoutError:
                logger.warning("Depth correction computation timed out; skipping this VP")
                continue
        if correction_factors:
            logger.debug("Computing mean correction factor")
            mean_correction = timeout_handler(np.mean, correction_factors, axis=0, timeout=5)
            enhanced_depth = enhanced_depth * mean_correction
            enhanced_depth = timeout_handler(cv2.bilateralFilter, enhanced_depth.astype(np.float32), d=9, sigmaColor=0.05, sigmaSpace=5, timeout=10)
        logger.info(f"VP-SLAM completed with correction in {time.time() - start_time:.2f}s")
        return enhanced_depth, vis_image
    else:
        logger.warning("No valid vanishing points found for VP-SLAM")
        logger.info(f"VP-SLAM completed without correction in {time.time() - start_time:.2f}s")
        return depth_map, vis_image

# === Mesh Generation Utilities ===
def depth_to_point_cloud(depth_map, f, cx, cy):
    """Convert a depth map to a 3D point cloud using a pinhole camera model."""
    logger.info("Converting depth map to point cloud")
    h, w = depth_map.shape
    i, j = timeout_handler(np.indices, (h, w), timeout=5)
    X = (j - cx) * depth_map / f
    Y = (i - cy) * depth_map / f
    Z = depth_map
    points = timeout_handler(np.stack, (X, Y, Z), axis=-1, timeout=5).reshape(-1, 3)
    logger.info("Point cloud generation completed")
    return points

def generate_mesh_from_depth(depth_map, f=DEFAULT_FOCAL):
    """Generate a mesh from a depth map by converting it to a point cloud."""
    logger.info("Generating mesh from depth map")
    start_time = time.time()
    h, w = depth_map.shape
    cx = w / 2.0
    cy = h / 2.0
    vertices = depth_to_point_cloud(depth_map, f, cx, cy)
    faces = []
    for i in range(h - 1):
        for j in range(w - 1):
            idx = i * w + j
            v0 = idx
            v1 = idx + 1
            v2 = idx + w
            v3 = idx + w + 1
            faces.append([v0, v1, v2])
            faces.append([v2, v1, v3])
    faces = timeout_handler(np.array, faces, timeout=5)
    logger.info(f"Mesh generated with {len(vertices)} vertices and {len(faces)} faces in {time.time() - start_time:.2f}s")
    return vertices, faces

def write_stl(vertices, faces, filename):
    """Write a triangle mesh (vertices, faces) to an STL file."""
    logger.info(f"Writing STL file to {filename}")
    ensure_dir(os.path.dirname(filename))
    start_time = time.time()
    if STL_AVAILABLE:
        stl_data = timeout_handler(np.zeros, (faces.shape[0],), dtype=stl_mesh.Mesh.dtype, timeout=5)
        for i, face in enumerate(faces):
            pts = vertices[face]
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            normal = timeout_handler(np.cross, v1, v2, timeout=1)
            norm = timeout_handler(np.linalg.norm, normal, timeout=1) + 1e-8
            normal /= norm
            stl_data["vectors"][i] = pts
            stl_data["normals"][i] = normal
        m = stl_mesh.Mesh(stl_data)
        timeout_handler(m.save, filename, timeout=10)
        logger.info(f"STL saved via numpy-stl in {time.time() - start_time:.2f}s")
    else:
        stl_str = "solid mesh\n"
        for face in faces:
            pts = vertices[face]
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            normal = timeout_handler(np.cross, v1, v2, timeout=1)
            norm = timeout_handler(np.linalg.norm, normal, timeout=1) + 1e-8
            normal /= norm
            stl_str += f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n"
            stl_str += "    outer loop\n"
            for pt in pts:
                stl_str += f"      vertex {pt[0]:.6e} {pt[1]:.6e} {pt[2]:.6e}\n"
            stl_str += "    endloop\n  endfacet\n"
        stl_str += "endsolid mesh\n"
        with open(filename, "w") as f:
            f.write(stl_str)
        logger.info(f"STL saved using basic writer in {time.time() - start_time:.2f}s")

# === Scene Understanding & Mesh Generation ===
def generate_scene_mesh(frame_path, depth_map):
    """Perform holistic scene understanding: mesh generation and segmentation."""
    logger.info(f"Generating scene mesh for: {frame_path}")
    start_time = time.time()
    if not os.path.exists(frame_path):
        logger.error(f"Frame file not found: {frame_path}")
        sys.exit(1)
    frame = timeout_handler(cv2.imread, frame_path, timeout=5)
    if frame is None:
        logger.error(f"Failed to load frame: {frame_path}")
        sys.exit(1)
    h, w = depth_map.shape
    vertices, faces = generate_mesh_from_depth(depth_map, f=DEFAULT_FOCAL)
    mesh_filename = os.path.join(MESH_DIR, f"mesh_{Path(frame_path).stem}.stl")
    write_stl(vertices, faces, mesh_filename)

    logger.info("Running instance segmentation")
    inst_model = load_instance_segmentation_model()
    frame_rgb = timeout_handler(cv2.cvtColor, frame, cv2.COLOR_BGR2RGB, timeout=5)
    transform_inst = Compose([torchvision.transforms.ToTensor()])
    input_tensor = transform_inst(frame_rgb).to(DEVICE)
    with torch.no_grad():
        predictions = timeout_handler(inst_model, [input_tensor], timeout=60)[0]
    seg_overlay = frame.copy()
    for i, mask in enumerate(predictions.get("masks", [])):
        mask = mask[0].mul(255).byte().cpu().numpy()
        color = timeout_handler(np.random.randint, 0, 255, (3,), dtype=np.uint8, timeout=1).tolist()
        seg_overlay[mask > 128] = (0.5 * seg_overlay[mask > 128] + 0.5 * np.array(color)).astype(np.uint8)
    overlay_filename = os.path.join(SCENE_DIR, f"segmentation_overlay_{Path(frame_path).stem}.png")
    ensure_dir(SCENE_DIR)
    timeout_handler(cv2.imwrite, overlay_filename, seg_overlay, timeout=5)
    logger.info(f"Scene processing completed in {time.time() - start_time:.2f}s: Mesh: {mesh_filename}, Overlay: {overlay_filename}")
    return mesh_filename, overlay_filename

# === Command-Line Interface ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ultra-Detailed 3D Scene Understanding, Mesh Generation, and Depth Fusion Pipeline"
    )
    parser.add_argument("--step", type=str,
                        choices=["frames", "depth", "fusion", "vpslam", "mesh", "scene"],
                        required=True, help="Pipeline step to execute")
    parser.add_argument("--frame_idx", type=int, default=None, help="Frame index for single-frame processing")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for depth estimation")
    parser.add_argument("--hough_threshold", type=int, default=150, help="Hough transform threshold for VP-SLAM")
    parser.add_argument("--max_iterations", type=int, default=5000, help="Max iterations for VP-SLAM vanishing point calculation")
    parser.add_argument("--profile", action="store_true", help="Enable profiling for detailed performance stats")
    args = parser.parse_args()

    logger.info(f"Starting pipeline with step: {args.step}, device: {DEVICE}, hough_threshold: {args.hough_threshold}, max_iterations: {args.max_iterations}, profile: {args.profile}")

    if args.step == "frames":
        extract_frames(VIDEO_PATH, FRAME_DIR)

    elif args.step == "depth":
        midas_model, midas_transform = load_midas_model(MODEL_PATH)
        ensure_dir(DEPTH_DIR)
        if args.frame_idx is not None:
            frame_file = os.path.join(FRAME_DIR, f"frame_{args.frame_idx:04d}.png")
            if not os.path.exists(frame_file):
                logger.error(f"Frame file not found: {frame_file}")
                sys.exit(1)
            depth = predict_depth(midas_model, midas_transform, [frame_file])[0]
            depth_file = os.path.join(DEPTH_DIR, f"depth_{args.frame_idx:04d}.png")
            save_depth_visual(depth, depth_file)
        else:
            frame_files = sorted([f for f in os.listdir(FRAME_DIR) if f.endswith(".png")])
            total_frames = len(frame_files)
            logger.info(f"Processing {total_frames} frames in batches of {args.batch_size}")
            for i in range(0, total_frames, args.batch_size):
                batch_paths = [os.path.join(FRAME_DIR, f) for f in frame_files[i:i+args.batch_size]]
                depth_maps = predict_depth(midas_model, midas_transform, batch_paths)
                for depth, fname in zip(depth_maps, batch_paths):
                    idx = int(Path(fname).stem.split("_")[1])
                    depth_file = os.path.join(DEPTH_DIR, f"depth_{idx:04d}.png")
                    save_depth_visual(depth, depth_file)

    elif args.step == "fusion":
        midas_model, midas_transform = load_midas_model(MODEL_PATH)
        ensure_dir(FUSION_DIR)
        if args.frame_idx is None:
            logger.error("Fusion step requires a specific frame index")
            sys.exit(1)
        frame_file = os.path.join(FRAME_DIR, f"frame_{args.frame_idx:04d}.png")
        depth_file = os.path.join(DEPTH_DIR, f"depth_{args.frame_idx:04d}.png")
        if not os.path.exists(frame_file):
            logger.error(f"Frame file not found: {frame_file}")
            sys.exit(1)
        if os.path.exists(depth_file):
            depth_map = plt.imread(depth_file)
            if len(depth_map.shape) == 3:
                depth_map = depth_map[:, :, 0]
        else:
            depth_map = predict_depth(midas_model, midas_transform, [frame_file])[0]
            save_depth_visual(depth_map, depth_file)
        fused_depth, semi_dense_depth, reliability_map = process_frame_with_deeprelativefusion(
            frame_file, depth_map, midas_model, midas_transform
        )
        fused_depth_file = os.path.join(FUSION_DIR, f"fused_depth_{args.frame_idx:04d}.png")
        plt.imsave(fused_depth_file, fused_depth, cmap="plasma")
        reliability_file = os.path.join(FUSION_DIR, f"reliability_{args.frame_idx:04d}.png")
        plt.imsave(reliability_file, reliability_map, cmap="viridis")
        logger.info(f"Fusion completed for frame {args.frame_idx}")

    elif args.step == "vpslam":
        midas_model, midas_transform = load_midas_model(MODEL_PATH)
        vp_dir = os.path.join(FUSION_DIR, "vp_slam")
        ensure_dir(vp_dir)
        if args.frame_idx is None:
            logger.error("VP-SLAM requires a specific frame index")
            sys.exit(1)
        frame_file = os.path.join(FRAME_DIR, f"frame_{args.frame_idx:04d}.png")
        depth_file = os.path.join(DEPTH_DIR, f"depth_{args.frame_idx:04d}.png")
        if not os.path.exists(frame_file):
            logger.error(f"Frame file not found: {frame_file}")
            sys.exit(1)
        if os.path.exists(depth_file):
            logger.info(f"Loading existing depth map: {depth_file}")
            depth_map = plt.imread(depth_file)
            if len(depth_map.shape) == 3:
                depth_map = depth_map[:, :, 0]
        else:
            logger.info(f"Generating depth map for frame: {frame_file}")
            depth_map = predict_depth(midas_model, midas_transform, [frame_file])[0]
            save_depth_visual(depth_map, depth_file)
        if args.profile:
            enhanced_depth, vp_vis = profile_function(
                process_frame_with_vp_slam, frame_file, depth_map,
                hough_threshold=args.hough_threshold, max_iterations=args.max_iterations
            )
        else:
            enhanced_depth, vp_vis = process_frame_with_vp_slam(
                frame_file, depth_map,
                hough_threshold=args.hough_threshold, max_iterations=args.max_iterations
            )
        enhanced_depth_file = os.path.join(vp_dir, f"vp_depth_{args.frame_idx:04d}.png")
        plt.imsave(enhanced_depth_file, enhanced_depth, cmap="plasma")
        vp_vis_file = os.path.join(vp_dir, f"vp_vis_{args.frame_idx:04d}.png")
        cv2.imwrite(vp_vis_file, vp_vis)
        logger.info(f"VP-SLAM processing completed for frame {args.frame_idx}")

    elif args.step == "mesh":
        ensure_dir(MESH_DIR)
        if args.frame_idx is None:
            logger.error("Mesh generation requires a specific frame index")
            sys.exit(1)
        frame_file = os.path.join(FRAME_DIR, f"frame_{args.frame_idx:04d}.png")
        depth_file = os.path.join(DEPTH_DIR, f"depth_{args.frame_idx:04d}.png")
        if not os.path.exists(frame_file):
            logger.error(f"Frame file not found: {frame_file}")
            sys.exit(1)
        if os.path.exists(depth_file):
            depth_map = plt.imread(depth_file)
            if len(depth_map.shape) == 3:
                depth_map = depth_map[:, :, 0]
        else:
            midas_model, midas_transform = load_midas_model(MODEL_PATH)
            depth_map = predict_depth(midas_model, midas_transform, [frame_file])[0]
            save_depth_visual(depth_map, depth_file)
        vertices, faces = generate_mesh_from_depth(depth_map, f=DEFAULT_FOCAL)
        mesh_filename = os.path.join(MESH_DIR, f"mesh_{args.frame_idx:04d}.stl")
        write_stl(vertices, faces, mesh_filename)
        logger.info(f"Mesh generation completed for frame {args.frame_idx}")

    elif args.step == "scene":
        ensure_dir(SCENE_DIR)
        if args.frame_idx is None:
            logger.error("Scene processing requires a specific frame index")
            sys.exit(1)
        frame_file = os.path.join(FRAME_DIR, f"frame_{args.frame_idx:04d}.png")
        depth_file = os.path.join(DEPTH_DIR, f"depth_{args.frame_idx:04d}.png")
        if not os.path.exists(frame_file):
            logger.error(f"Frame file not found: {frame_file}")
            sys.exit(1)
        if os.path.exists(depth_file):
            depth_map = plt.imread(depth_file)
            if len(depth_map.shape) == 3:
                depth_map = depth_map[:, :, 0]
        else:
            midas_model, midas_transform = load_midas_model(MODEL_PATH)
            depth_map = predict_depth(midas_model, midas_transform, [frame_file])[0]
            save_depth_visual(depth_map, depth_file)
        mesh_filename, overlay_filename = generate_scene_mesh(frame_file, depth_map)
        logger.info(f"Scene processing completed for frame {args.frame_idx}")
