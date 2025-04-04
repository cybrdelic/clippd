import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from skimage import measure
from pathlib import Path

def analyze_object_depths(frame_path, depth_path, mask_path):
    """
    Analyze the depth properties of each segmented object in the scene.

    Args:
        frame_path (str): Path to the original frame image
        depth_path (str): Path to the depth map
        mask_path (str): Path to the segmentation mask

    Returns:
        list: List of dictionaries containing analysis for each object
    """
    # Load the frame, depth map, and segmentation mask
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Load depth map (using matplotlib to maintain colormap if present)
    depth = plt.imread(depth_path)
    if depth.ndim == 3 and depth.shape[2] == 4:  # RGBA
        # Convert to grayscale (taking first channel)
        depth = depth[:,:,0]
    elif depth.ndim == 3 and depth.shape[2] == 3:  # RGB
        # Convert to grayscale
        depth = np.mean(depth, axis=2)

    # Load mask
    mask = plt.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[:,:,0]  # Get first channel

    # Find connected components in segmentation mask
    labeled_mask = measure.label(mask > 0.5)
    regions = measure.regionprops(labeled_mask)

    object_analysis = []

    # Calculate global depth statistics for relative positioning
    global_depth_median = np.median(depth)

    # Analyze each object
    for region in regions:
        if region.area < 100:  # Skip tiny regions
            continue

        # Create mask for this specific object
        object_mask = labeled_mask == region.label

        # Extract depth values for this object
        object_depths = depth[object_mask]

        # Calculate depth statistics
        analysis = {
            "id": region.label,
            "centroid": region.centroid,
            "bbox": region.bbox,
            "area": region.area,
            "mean_depth": float(np.mean(object_depths)),
            "min_depth": float(np.min(object_depths)),
            "max_depth": float(np.max(object_depths)),
            "depth_variance": float(np.var(object_depths)),
            "relative_position": "foreground" if np.mean(object_depths) < global_depth_median else "background"
        }

        object_analysis.append(analysis)

    # Sort objects by mean depth (front to back)
    object_analysis.sort(key=lambda x: x["mean_depth"])

    return object_analysis

def create_semantic_depth_visualization(frame_path, depth_path, mask_path, object_analysis, output_path=None):
    """
    Create a visualization that combines the frame, depth map, and object information.

    Args:
        frame_path (str): Path to the original frame image
        depth_path (str): Path to the depth map
        mask_path (str): Path to the segmentation mask
        object_analysis (list): List of object analysis dictionaries
        output_path (str, optional): Path to save the visualization image
    """
    # Load the frame, depth map, and segmentation mask
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Load depth map (using matplotlib to maintain colormap if present)
    depth = plt.imread(depth_path)

    # Create a blended background (frame + depth)
    visualization = frame.copy().astype(float) / 255.0

    # Apply depth as overlay with colored map
    if depth.ndim == 3 and depth.shape[2] == 4:  # RGBA
        depth_colored = depth[:,:,:3]  # Use the first 3 channels
    elif depth.ndim == 2:
        # Apply colormap to grayscale depth
        depth_colored = plt.cm.plasma(plt.Normalize()(depth))[:,:,:3]
    else:
        depth_colored = depth[:,:,:3]

    # Blend frame with depth
    for i in range(3):  # RGB channels
        visualization[:,:,i] = visualization[:,:,i] * 0.7 + depth_colored[:,:,i] * 0.3

    # Draw object annotations
    for obj in object_analysis:
        # Get object centroid
        y, x = int(obj["centroid"][0]), int(obj["centroid"][1])

        # Get object bounding box
        y1, x1, y2, x2 = obj["bbox"]

        # Determine color based on relative position
        if obj["relative_position"] == "foreground":
            box_color = (1.0, 0.2, 0.2)  # Red for foreground
        else:
            box_color = (0.2, 0.2, 1.0)  # Blue for background

        # Draw bounding box
        thickness = 2
        visualization[y1:y1+thickness, x1:x2] = box_color  # Top
        visualization[y2-thickness:y2, x1:x2] = box_color  # Bottom
        visualization[y1:y2, x1:x1+thickness] = box_color  # Left
        visualization[y1:y2, x2-thickness:x2] = box_color  # Right

        # Prepare label text (convert to uint8 to use OpenCV text)
        vis_uint8 = (visualization * 255).astype(np.uint8)

        # Draw a background for the text
        label = f"ID:{obj['id']} D:{obj['mean_depth']:.2f}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(
            vis_uint8,
            (x - 5, y - text_size[1] - 10),
            (x + text_size[0] + 5, y),
            (255, 255, 255),
            -1
        )

        # Add text
        cv2.putText(
            vis_uint8,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

        visualization = vis_uint8.astype(float) / 255.0

    # Convert to uint8 for saving
    visualization = (visualization * 255).astype(np.uint8)

    # Save the visualization if requested
    if output_path:
        plt.imsave(output_path, visualization)
        print(f"Saved semantic visualization to {output_path}")

    return visualization

def process_frame(frame_idx, base_dir="../data"):
    """Process a single frame, analyzing objects and creating visualization."""
    # Define paths
    frame_path = os.path.join(base_dir, "frames", f"frame_{frame_idx:04d}.png")
    depth_path = os.path.join(base_dir, "depth", f"depth_{frame_idx:04d}.png")
    mask_path = os.path.join(base_dir, "segmentation_masks", f"mask_{frame_idx:04d}.png")

    # Create output directories
    analysis_dir = os.path.join(base_dir, "analysis")
    viz_dir = os.path.join(base_dir, "semantic_viz")
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    # Define output paths
    analysis_path = os.path.join(analysis_dir, f"analysis_{frame_idx:04d}.json")
    viz_path = os.path.join(viz_dir, f"semantic_{frame_idx:04d}.png")

    # Check if input files exist
    if not all(os.path.exists(p) for p in [frame_path, depth_path, mask_path]):
        print(f"Error: Missing files for frame {frame_idx}")
        return False

    # Analyze objects
    try:
        object_analysis = analyze_object_depths(frame_path, depth_path, mask_path)

        # Save analysis to JSON
        import json
        with open(analysis_path, 'w') as f:
            # Convert NumPy types to native Python types for JSON serialization
            serializable_analysis = []
            for obj in object_analysis:
                obj_copy = obj.copy()
                obj_copy["centroid"] = (float(obj["centroid"][0]), float(obj["centroid"][1]))
                obj_copy["bbox"] = tuple(int(x) for x in obj["bbox"])
                serializable_analysis.append(obj_copy)

            json.dump(serializable_analysis, f, indent=2)

        # Create visualization
        create_semantic_depth_visualization(frame_path, depth_path, mask_path, object_analysis, viz_path)

        print(f"Processed frame {frame_idx}: Found {len(object_analysis)} objects")
        return True

    except Exception as e:
        print(f"Error processing frame {frame_idx}: {e}")
        return False

def main():
    """Main function to parse arguments and run analysis."""
    parser = argparse.ArgumentParser(description="Advanced depth and segmentation analysis")
    parser.add_argument("--base_dir", type=str, default="../data", help="Base directory for data")
    parser.add_argument("--frame_idx", type=int, default=None, help="Process a specific frame")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame index")

    args = parser.parse_args()

    if args.frame_idx is not None:
        # Process a single frame
        process_frame(args.frame_idx, args.base_dir)
    else:
        # Process multiple frames
        frame_dir = os.path.join(args.base_dir, "frames")
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])

        if args.end_frame is None:
            args.end_frame = len(frame_files)

        for i in range(args.start_frame, min(args.end_frame, len(frame_files))):
            frame_idx = int(frame_files[i].split("_")[1].split(".")[0])
            if process_frame(frame_idx, args.base_dir) and (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{min(args.end_frame, len(frame_files))}")

if __name__ == "__main__":
    main()
