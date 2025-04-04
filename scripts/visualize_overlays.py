import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
import cv2
import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def create_interactive_overlay_visualization(base_dir="../data"):
    # Get list of available frames
    frame_dir = os.path.join(base_dir, "frames")
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])
    total_frames = len(frame_files)

    if total_frames == 0:
        print(f"No frames found in {frame_dir}")
        return

    # Create figure and subplots
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.25)  # Make room for controls

    # Create slider axis and widget
    ax_frame = plt.axes([0.15, 0.15, 0.7, 0.03])
    frame_slider = Slider(
        ax=ax_frame,
        label='Frame',
        valmin=0,
        valmax=total_frames-1,
        valinit=0,
        valstep=1
    )

    # Create depth opacity slider
    ax_depth = plt.axes([0.15, 0.10, 0.7, 0.03])
    depth_slider = Slider(
        ax=ax_depth,
        label='Depth Opacity',
        valmin=0.0,
        valmax=1.0,
        valinit=0.4,
        valstep=0.05
    )

    # Create mask opacity slider
    ax_mask = plt.axes([0.15, 0.05, 0.7, 0.03])
    mask_slider = Slider(
        ax=ax_mask,
        label='Mask Opacity',
        valmin=0.0,
        valmax=1.0,
        valinit=0.3,
        valstep=0.05
    )

    # Create checkboxes for visibility
    ax_check = plt.axes([0.05, 0.35, 0.1, 0.15])
    check = CheckButtons(
        ax=ax_check,
        labels=['Frame', 'Depth', 'Mask'],
        actives=[True, True, True]
    )

    # Create button for saving the current view
    ax_save = plt.axes([0.8, 0.05, 0.15, 0.05])
    save_button = Button(ax_save, 'Save Image')

    # Initialize state variables
    current_frame = 0
    show_frame = True
    show_depth = True
    show_mask = True

    # Set title
    plt.suptitle('Frame, Depth, and Segmentation Visualization', fontsize=16)

    def update_display():
        ax.clear()

        # Get frame index
        frame_idx = int(current_frame)

        # Load images
        frame_path = os.path.join(base_dir, "frames", f"frame_{frame_idx:04d}.png")
        depth_path = os.path.join(base_dir, "depth", f"depth_{frame_idx:04d}.png")
        mask_path = os.path.join(base_dir, "segmentation_masks", f"mask_{frame_idx:04d}.png")

        # Check if files exist
        if not os.path.exists(frame_path):
            ax.text(0.5, 0.5, f"Frame {frame_idx} not found", ha='center', va='center')
            fig.canvas.draw_idle()
            return

        # Read images
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Start with a black background
        composite = np.zeros_like(frame, dtype=float)

        # Add frame if visible
        if show_frame:
            composite = frame.copy().astype(float) / 255.0

        # Add depth if visible and file exists
        if show_depth and os.path.exists(depth_path):
            depth = plt.imread(depth_path)

            # If depth is grayscale, convert to colormap
            if depth.ndim == 2 or (depth.ndim == 3 and depth.shape[2] == 1):
                depth_colored = plt.cm.plasma(plt.Normalize()(depth))
            else:
                depth_colored = depth

            # Apply depth with opacity
            depth_opacity = depth_slider.val
            if show_frame:
                # Blend with frame
                for i in range(3):  # RGB channels
                    composite[:,:,i] = composite[:,:,i] * (1 - depth_opacity) + depth_colored[:,:,i] * depth_opacity
            else:
                # Show only depth
                composite = depth_colored[:,:,:3] * depth_opacity

        # Add mask if visible and file exists
        if show_mask and os.path.exists(mask_path):
            mask = plt.imread(mask_path)

            # If mask is grayscale with values near 0 and 1, use binary threshold
            if mask.ndim == 2 or (mask.ndim == 3 and mask.shape[2] == 1):
                mask_binary = mask > 0.5
            elif mask.shape[2] == 4:  # RGBA
                mask_binary = mask[:,:,0] > 0.5
            else:
                mask_binary = np.mean(mask, axis=2) > 0.5

            # Create colored overlay for mask
            mask_overlay = np.zeros_like(frame, dtype=float)
            mask_color = [1.0, 0.2, 0.2]  # Red color for mask

            for i in range(3):
                mask_overlay[:,:,i] = mask_binary * mask_color[i]

            # Apply mask with opacity
            mask_opacity = mask_slider.val
            for i in range(3):
                composite[:,:,i] = composite[:,:,i] * (1 - mask_binary * mask_opacity) + mask_overlay[:,:,i] * mask_opacity

        # Display composite image
        ax.imshow(composite)
        ax.set_title(f"Frame {frame_idx}")
        ax.axis('off')

        # Add frame info text
        ax.text(0.02, 0.98, f"Frame: {frame_idx}/{total_frames-1}",
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Redraw canvas
        fig.canvas.draw_idle()

    def update_frame(val):
        nonlocal current_frame
        current_frame = int(val)
        update_display()

    def update_visibility(label):
        nonlocal show_frame, show_depth, show_mask
        if label == 'Frame':
            show_frame = not show_frame
        elif label == 'Depth':
            show_depth = not show_depth
        elif label == 'Mask':
            show_mask = not show_mask
        update_display()

    def save_current_view(event):
        frame_idx = int(current_frame)
        save_dir = os.path.join(base_dir, "visualizations")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"visualization_{frame_idx:04d}.png")

        # Get current figure as image
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved visualization to {save_path}")

    # Connect callbacks
    frame_slider.on_changed(update_frame)
    depth_slider.on_changed(lambda val: update_display())
    mask_slider.on_changed(lambda val: update_display())
    check.on_clicked(update_visibility)
    save_button.on_clicked(save_current_view)

    # Add keyboard navigation
    def key_press(event):
        nonlocal current_frame
        if event.key == 'right' or event.key == 'n':
            current_frame = min(current_frame + 1, total_frames - 1)
            frame_slider.set_val(current_frame)
        elif event.key == 'left' or event.key == 'p':
            current_frame = max(current_frame - 1, 0)
            frame_slider.set_val(current_frame)
        elif event.key == 'home':
            current_frame = 0
            frame_slider.set_val(current_frame)
        elif event.key == 'end':
            current_frame = total_frames - 1
            frame_slider.set_val(current_frame)

    fig.canvas.mpl_connect('key_press_event', key_press)

    # Initial display
    update_display()

    plt.show()

def create_video_from_visualization(base_dir="../data", output_path="../data/visualization.mp4", fps=30,
                                   depth_opacity=0.4, mask_opacity=0.3):
    """Create a video file from the visualization"""
    # Get list of available frames
    frame_dir = os.path.join(base_dir, "frames")
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])
    total_frames = len(frame_files)

    if total_frames == 0:
        print(f"No frames found in {frame_dir}")
        return

    # Get dimensions from first frame
    first_frame_path = os.path.join(frame_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, _ = first_frame.shape

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Creating video with {total_frames} frames...")

    # Process each frame
    for i, frame_file in enumerate(frame_files):
        frame_idx = int(frame_file.split('_')[1].split('.')[0])

        # Load images
        frame_path = os.path.join(base_dir, "frames", f"frame_{frame_idx:04d}.png")
        depth_path = os.path.join(base_dir, "depth", f"depth_{frame_idx:04d}.png")
        mask_path = os.path.join(base_dir, "segmentation_masks", f"mask_{frame_idx:04d}.png")

        # Check if all files exist
        if not os.path.exists(frame_path):
            print(f"Skipping frame {frame_idx}, missing frame file")
            continue

        # Read images
        frame = cv2.imread(frame_path)

        # Start with original frame
        composite = frame.copy().astype(float)

        # Add depth if file exists
        if os.path.exists(depth_path):
            # Need to convert the depth visualization to BGR for OpenCV
            depth_rgb = cv2.imread(depth_path)

            # Apply depth with opacity
            composite = composite * (1 - depth_opacity) + depth_rgb * depth_opacity

        # Add mask if file exists
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_binary = mask > 128

            # Create red overlay for mask
            mask_overlay = np.zeros_like(frame, dtype=float)
            mask_overlay[mask_binary] = [0, 0, 255]  # Red in BGR

            # Apply mask with opacity
            for i in range(3):
                composite[:,:,i] = composite[:,:,i] * (1 - mask_binary * mask_opacity) + mask_overlay[:,:,i] * mask_opacity

        # Convert to uint8 for video
        composite = np.clip(composite, 0, 255).astype(np.uint8)

        # Add frame number
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(composite, f'Frame: {frame_idx}', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Write to video
        video.write(composite)

        if i % 10 == 0:
            print(f"Processed {i}/{total_frames} frames")

    # Release video writer
    video.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    # Run the interactive visualization
    create_interactive_overlay_visualization()

    # Uncomment to create a video instead:
    # create_video_from_visualization()
