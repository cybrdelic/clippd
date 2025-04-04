import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from skimage import measure
from scipy.spatial import distance
from scipy.optimize import curve_fit, linear_sum_assignment

# Add necessary directories from your existing setup
sys.path.append("C:/Users/alexf/software-projects/clippd/midas")

# === Configuration ===
BASE_DIR = "../data"
FRAME_DIR = os.path.join(BASE_DIR, "frames")
DEPTH_DIR = os.path.join(BASE_DIR, "depth")
SEGMENTATION_DIR = os.path.join(BASE_DIR, "segmentation_masks")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")
PHYSICS_DIR = os.path.join(BASE_DIR, "physics_viz")
FRAME_RATE = 30  # Default frame rate for game footage

# === Object Tracking ===
def track_objects_between_frames(prev_objects, curr_objects, max_distance=50):
    """Track objects across consecutive frames based on spatial proximity and appearance."""
    if not prev_objects:
        # First frame - assign new IDs
        for i, obj in enumerate(curr_objects):
            obj["tracking_id"] = i
        return curr_objects

    # Create cost matrix for assignment
    cost_matrix = np.zeros((len(prev_objects), len(curr_objects)))
    for i, prev_obj in enumerate(prev_objects):
        for j, curr_obj in enumerate(curr_objects):
            # Calculate centroid distance
            prev_y, prev_x = prev_obj["centroid"]
            curr_y, curr_x = curr_obj["centroid"]
            dist = np.sqrt((prev_y - curr_y)**2 + (prev_x - curr_x)**2)

            # Calculate depth similarity
            depth_diff = abs(prev_obj["mean_depth"] - curr_obj["mean_depth"])

            # Calculate area similarity
            area_ratio = min(prev_obj["area"], curr_obj["area"]) / max(prev_obj["area"], curr_obj["area"])

            # Combined cost (lower is better)
            cost_matrix[i, j] = dist * (1 + depth_diff) * (2 - area_ratio)

    # Assign tracking IDs using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Apply assignments where cost is reasonable
    assigned = set()
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < max_distance:
            curr_objects[j]["tracking_id"] = prev_objects[i]["tracking_id"]
            assigned.add(j)

    # Assign new IDs to unassigned objects
    next_id = max([obj.get("tracking_id", -1) for obj in prev_objects], default=-1) + 1
    for j, obj in enumerate(curr_objects):
        if j not in assigned:
            obj["tracking_id"] = next_id
            next_id += 1

    return curr_objects

# === Physics Analysis ===
def analyze_physics(object_history, frame_rate=FRAME_RATE):
    """Analyze physics properties from object tracking history."""
    physics_data = {}

    # Process each tracked object
    for tracking_id, frames in object_history.items():
        # Need at least 3 frames for acceleration analysis
        if len(frames) < 3:
            continue

        # Extract position data
        positions = []
        times = []
        for frame_idx, obj_data in frames:
            y, x = obj_data["centroid"]
            positions.append((x, y))  # Note: x, y format for plotting
            times.append(frame_idx / frame_rate)  # Convert to seconds

        # Calculate velocities (pixels per second)
        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            dt = times[i] - times[i-1]
            if dt > 0:
                vx = dx / dt
                vy = dy / dt
                velocities.append((vx, vy))
            else:
                velocities.append((0, 0))

        # Calculate accelerations
        accelerations = []
        for i in range(1, len(velocities)):
            dvx = velocities[i][0] - velocities[i-1][0]
            dvy = velocities[i][1] - velocities[i-1][1]
            dt = times[i] - times[i-1]
            if dt > 0:
                ax = dvx / dt
                ay = dvy / dt
                accelerations.append((ax, ay))
            else:
                accelerations.append((0, 0))

        # Estimate trajectory type
        trajectory_type = "unknown"
        trajectory_params = {}

        if len(positions) >= 5:
            # Check if it's a projectile motion
            try:
                # Separate x and y coordinates
                xs, ys = zip(*positions)
                xs, ys = np.array(xs), np.array(ys)

                # Try fitting a parabola (y = a*x^2 + b*x + c)
                def parabola(x, a, b, c):
                    return a * (x**2) + b * x + c

                # Normalize x values for better fitting
                x_norm = np.array(xs) - xs[0]
                params, covariance = curve_fit(parabola, x_norm, ys)

                # If 'a' parameter is significant, it's likely a projectile
                if abs(params[0]) > 0.001:
                    trajectory_type = "projectile"
                    # Estimate gravity (pixel/s^2)
                    gravity_estimate = 2 * params[0] * (frame_rate**2)
                    trajectory_params = {
                        "a": float(params[0]),
                        "b": float(params[1]),
                        "c": float(params[2]),
                        "gravity": float(gravity_estimate)
                    }
                else:
                    # Check if it's linear motion
                    avg_acceleration = 0
                    if accelerations:
                        avg_acceleration = np.mean([np.sqrt(ax**2 + ay**2) for ax, ay in accelerations])

                    if avg_acceleration < 50:  # Threshold for "constant velocity"
                        trajectory_type = "linear"
                        # Linear fit (y = mx + b)
                        def linear(x, m, b):
                            return m * x + b

                        try:
                            params, _ = curve_fit(linear, x_norm, ys)
                            trajectory_params = {
                                "slope": float(params[0]),
                                "intercept": float(params[1])
                            }
                        except:
                            trajectory_params = {}
                    else:
                        trajectory_type = "complex"
            except Exception as e:
                print(f"Error fitting trajectory for object {tracking_id}: {e}")
                trajectory_type = "complex"

        # Calculate average speed
        avg_speed = 0
        if velocities:
            avg_speed = np.mean([np.sqrt(vx**2 + vy**2) for vx, vy in velocities])

        # Store physics data
        physics_data[tracking_id] = {
            "positions": positions,
            "times": times,
            "velocities": velocities,
            "accelerations": accelerations,
            "trajectory_type": trajectory_type,
            "trajectory_params": trajectory_params,
            "last_position": positions[-1] if positions else (0, 0),
            "last_velocity": velocities[-1] if velocities else (0, 0),
            "avg_speed": float(avg_speed),
        }

        # Predict future positions (next 10 frames)
        future_positions = []
        if trajectory_type == "projectile" and trajectory_params and len(positions) >= 3:
            # Use parabolic trajectory for prediction
            x0, y0 = positions[-1]
            vx, vy = velocities[-1] if velocities else (0, 0)

            # Use fitted parameters for prediction
            a, b, c = trajectory_params["a"], trajectory_params["b"], trajectory_params["c"]

            for i in range(1, 11):
                t = i / frame_rate
                # Calculate future x position assuming constant x velocity
                x = x0 + vx * t
                # Calculate y using parabolic fit
                x_norm = (x - positions[0][0])
                y = a * (x_norm**2) + b * x_norm + c
                future_positions.append((x, y))

        elif trajectory_type == "linear" and trajectory_params and velocities:
            # Use linear trajectory for prediction
            x0, y0 = positions[-1]
            vx, vy = velocities[-1]

            for i in range(1, 11):
                t = i / frame_rate
                x = x0 + vx * t
                y = y0 + vy * t
                future_positions.append((x, y))

        physics_data[tracking_id]["future_positions"] = future_positions

    return physics_data

# === Game Physics Analysis ===
def analyze_game_physics(objects, physics_data, depth_map):
    """Analyze game-specific physics like bullet trajectories, player movement, etc."""
    game_physics = {
        "projectiles": [],
        "players": [],
        "structures": [],
        "gravity_estimate": None,
    }

    # Identify projectiles (bullets, rockets, etc.)
    projectiles = [obj["tracking_id"] for obj in objects if
                  obj["tracking_id"] in physics_data and
                  physics_data[obj["tracking_id"]]["trajectory_type"] == "projectile" and
                  physics_data[obj["tracking_id"]]["avg_speed"] > 100]  # Fast moving objects

    if projectiles:
        # Estimate global gravity from all projectiles
        gravity_values = [physics_data[tid]["trajectory_params"].get("gravity", 0)
                         for tid in projectiles
                         if "gravity" in physics_data[tid]["trajectory_params"]]

        if gravity_values:
            game_physics["gravity_estimate"] = float(np.median(gravity_values))

        # Analyze each projectile
        for tid in projectiles:
            physics = physics_data[tid]
            obj = next((o for o in objects if o["tracking_id"] == tid), None)
            if not obj:
                continue

            # Calculate projectile data
            projectile_data = {
                "id": tid,
                "initial_position": physics["positions"][0] if physics["positions"] else (0, 0),
                "current_position": physics["positions"][-1] if physics["positions"] else (0, 0),
                "velocity": physics["last_velocity"],
                "speed": physics["avg_speed"],
                "trajectory_type": physics["trajectory_type"],
            }

            # Calculate bullet drop if applicable
            if len(physics["positions"]) > 2:
                start_x, start_y = physics["positions"][0]
                end_x, end_y = physics["positions"][-1]
                horizontal_dist = abs(end_x - start_x)
                vertical_drop = end_y - start_y

                projectile_data["horizontal_distance"] = float(horizontal_dist)
                projectile_data["vertical_drop"] = float(vertical_drop)

                # Calculate drop ratio (drop per distance traveled)
                if horizontal_dist > 0:
                    projectile_data["drop_ratio"] = float(vertical_drop / horizontal_dist)

            game_physics["projectiles"].append(projectile_data)

    # Identify players by size and movement patterns
    player_ids = [obj["tracking_id"] for obj in objects if
                 obj["area"] > 3000 and  # Large objects are likely players
                 obj["tracking_id"] in physics_data]

    for tid in player_ids:
        physics = physics_data[tid]
        obj = next((o for o in objects if o["tracking_id"] == tid), None)
        if not obj:
            continue

        # Calculate player movement data
        player_data = {
            "id": tid,
            "position": physics["positions"][-1] if physics["positions"] else (0, 0),
            "velocity": physics["last_velocity"],
            "speed": physics["avg_speed"],
            "depth": float(obj["mean_depth"]),
        }

        # Determine if player is jumping (vertical acceleration)
        if len(physics["accelerations"]) > 2:
            vertical_accel = [a[1] for a in physics["accelerations"][-3:]]
            avg_vert_accel = np.mean(vertical_accel)
            player_data["vertical_acceleration"] = float(avg_vert_accel)
            player_data["is_jumping"] = avg_vert_accel < -50  # Negative y is up in image coordinates

        game_physics["players"].append(player_data)

    # Identify static structures (buildings, terrain, etc.)
    structure_ids = [obj["tracking_id"] for obj in objects if
                    obj["area"] > 1000 and
                    obj["tracking_id"] in physics_data and
                    physics_data[obj["tracking_id"]]["avg_speed"] < 10]  # Slow/static objects

    for tid in structure_ids:
        obj = next((o for o in objects if o["tracking_id"] == tid), None)
        if not obj:
            continue

        structure_data = {
            "id": tid,
            "position": (float(obj["centroid"][1]), float(obj["centroid"][0])),  # x, y format
            "size": float(obj["area"]),
            "depth": float(obj["mean_depth"]),
        }

        game_physics["structures"].append(structure_data)

    return game_physics

# === Physics Visualization ===
def create_physics_visualization(frame, objects, physics_data, game_physics=None, depth_map=None, output_path=None):
    """Create visualization with physics information overlays."""
    # Create a copy for visualization
    visualization = frame.copy()

    # Blend with depth map if provided
    if depth_map is not None:
        # Ensure depth_map is in proper format for visualization
        if depth_map.ndim == 2:
            # Convert grayscale depth to color visualization
            depth_colored = cv2.applyColorMap(
                cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_PLASMA
            )
        elif depth_map.ndim == 3 and depth_map.shape[2] == 3:
            # Already RGB
            depth_colored = depth_map.copy()
        else:
            # Other formats - try to convert
            try:
                # For RGBA (like from matplotlib)
                if depth_map.ndim == 3 and depth_map.shape[2] == 4:
                    depth_gray = depth_map[:,:,0]  # Take first channel
                else:
                    depth_gray = np.mean(depth_map, axis=2)

                depth_colored = cv2.applyColorMap(
                    cv2.normalize(depth_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                    cv2.COLORMAP_PLASMA
                )
            except Exception as e:
                print(f"Error converting depth map for visualization: {e}")
                depth_colored = None

        if depth_colored is not None:
            try:
                # Ensure same dimensions as frame
                if depth_colored.shape[:2] != frame.shape[:2]:
                    depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
                visualization = cv2.addWeighted(visualization, 0.7, depth_colored, 0.3, 0)
            except Exception as e:
                print(f"Error blending depth with frame: {e}")

    # Add a semi-transparent overlay for drawing math
    math_overlay = np.zeros_like(visualization, dtype=np.uint8)

    # Draw physics information for each object
    for obj in objects:
        if obj["tracking_id"] not in physics_data:
            continue

        physics = physics_data[obj["tracking_id"]]
        y, x = obj["centroid"]
        y, x = int(y), int(x)

        # Color coding based on trajectory type
        if physics["trajectory_type"] == "projectile":
            color = (0, 0, 255)  # Red for projectiles
        elif physics["trajectory_type"] == "linear":
            color = (0, 255, 0)  # Green for linear motion
        else:
            color = (255, 0, 0)  # Blue for complex motion

        # Draw bounding box
        y1, x1, y2, x2 = obj["bbox"]
        cv2.rectangle(visualization, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

        # Draw velocity vector
        if physics["velocities"]:
            vx, vy = physics["last_velocity"]
            speed = np.sqrt(vx**2 + vy**2)

            # Skip drawing vectors for near-stationary objects
            if speed > 10:
                # Scale vector for visualization
                scale = min(50, max(5, speed / 5))
                end_x = int(x + vx * scale / speed) if speed > 0 else x
                end_y = int(y + vy * scale / speed) if speed > 0 else y

                # Make sure endpoints are within image bounds
                h, w = visualization.shape[:2]
                end_x = min(max(end_x, 0), w-1)
                end_y = min(max(end_y, 0), h-1)

                cv2.arrowedLine(visualization, (x, y), (end_x, end_y), color, 2)

                # Add speed label
                cv2.putText(
                    visualization,
                    f"v={speed:.1f}",
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

        # Draw trajectory prediction
        if "future_positions" in physics and physics["future_positions"]:
            points = [(int(px), int(py)) for px, py in physics["future_positions"]]

            # Filter out points outside image bounds
            h, w = visualization.shape[:2]
            valid_points = [(px, py) for px, py in points if 0 <= px < w and 0 <= py < h]

            # Draw predicted path
            prev_point = (x, y)
            for i, point in enumerate(valid_points):
                alpha = 1.0 - (i / (len(valid_points) + 1))  # Fade out future points
                line_color = tuple(int(c * alpha) for c in color)
                cv2.line(visualization, prev_point, point, line_color, 2)
                prev_point = point

            # Draw endpoint marker
            if valid_points:
                last_x, last_y = valid_points[-1]
                cv2.circle(visualization, (last_x, last_y), 3, color, -1)

    # Add mathematical formulas based on what's detected in the scene
    text_y = 30
    text_spacing = 30

    # Add title
    cv2.putText(
        visualization, "Physics Analysis", (20, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    text_y += text_spacing

    # Add formulas based on game physics
    if game_physics:
        # If projectiles detected, show projectile motion formula
        if game_physics["projectiles"]:
            # Draw projectile motion equation
            cv2.putText(
                visualization, "Projectile: y = y0 + v0*t + 1/2*g*t^2", (20, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1
            )
            text_y += text_spacing

            # Show estimated gravity if available
            if game_physics["gravity_estimate"]:
                cv2.putText(
                    visualization, f"g = {game_physics['gravity_estimate']:.1f} px/s^2", (20, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1
                )
                text_y += text_spacing

        # If players detected, show movement physics
        if game_physics["players"]:
            cv2.putText(
                visualization, "Movement: F = m*a", (20, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1
            )
            text_y += text_spacing

            # Add player speed
            fastest_player = max(game_physics["players"], key=lambda p: p["speed"], default=None)
            if fastest_player:
                cv2.putText(
                    visualization, f"Top Speed: {fastest_player['speed']:.1f} px/s", (20, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1
                )
                text_y += text_spacing

    # Add game-specific overlays
    if game_physics:
        # Highlight projectiles (bullets, rockets)
        for projectile in game_physics["projectiles"]:
            # Find object with this tracking ID
            obj = next((o for o in objects if o["tracking_id"] == projectile["id"]), None)
            if not obj:
                continue

            x, y = int(projectile["current_position"][0]), int(projectile["current_position"][1])

            # Draw bullet trajectory with drop indicator
            if "vertical_drop" in projectile and "horizontal_distance" in projectile:
                # Draw bullet drop visualization
                if abs(projectile["horizontal_distance"]) > 50:  # Only for bullets that traveled some distance
                    drop_text = f"Drop: {projectile['vertical_drop']:.1f} px"
                    cv2.putText(
                        visualization, drop_text, (x + 15, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
                    )

        # Add targeting info for players
        for player in game_physics["players"]:
            # Find object with this tracking ID
            obj = next((o for o in objects if o["tracking_id"] == player["id"]), None)
            if not obj:
                continue

            # Draw crosshair on player
            x, y = int(player["position"][0]), int(player["position"][1])
            radius = int(np.sqrt(obj["area"]) / 4)  # Size based on player size

            # Add player indicators
            if player.get("is_jumping", False):
                # Jumping animation
                cv2.putText(
                    visualization, "↑JUMP↑", (x - 30, y - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
                )

            # Add distance indicator based on depth
            distance_text = f"d={player['depth']:.1f}m"
            cv2.putText(
                visualization, distance_text, (x - 30, y + radius + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
            )

    # Save if requested
    if output_path:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save image
            cv2.imwrite(output_path, visualization)

            # Verify file was created
            if os.path.exists(output_path):
                print(f"Saved physics visualization to {output_path}")
            else:
                print(f"Failed to save to {output_path} (file not created)")
        except Exception as e:
            print(f"Error saving visualization: {e}")

    return visualization

# === Main Processing Functions ===
def load_frame_data(frame_idx, base_dir=BASE_DIR):
    """Load frame, depth, and analysis data for a given frame index."""
    frame_path = os.path.join(base_dir, "frames", f"frame_{frame_idx:04d}.png")
    depth_path = os.path.join(base_dir, "depth", f"depth_{frame_idx:04d}.png")
    mask_path = os.path.join(base_dir, "segmentation_masks", f"mask_{frame_idx:04d}.png")
    analysis_path = os.path.join(base_dir, "analysis", f"analysis_{frame_idx:04d}.json")

    # Debug: Check if files exist
    missing_files = []
    if not os.path.exists(frame_path):
        missing_files.append(f"Frame: {frame_path}")
    if not os.path.exists(depth_path):
        missing_files.append(f"Depth: {depth_path}")
    if not os.path.exists(analysis_path):
        missing_files.append(f"Analysis: {analysis_path}")

    if missing_files:
        print(f"For frame {frame_idx}, missing required files:")
        for missing in missing_files:
            print(f"  - {missing}")
        return None, None, None, None

    # Load frame
    try:
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Error: Failed to read frame image at {frame_path}")
            return None, None, None, None
    except Exception as e:
        print(f"Error loading frame for frame {frame_idx}: {e}")
        return None, None, None, None

    # Load depth (using matplotlib to maintain colormap if present)
    try:
        depth = plt.imread(depth_path)
        # If depth is RGBA, convert to grayscale
        if depth.ndim == 3 and depth.shape[2] == 4:
            depth = depth[:,:,0]  # Just take first channel
    except Exception as e:
        print(f"Error loading depth for frame {frame_idx}: {e}")
        depth = None

    # Load mask
    try:
        mask = plt.imread(mask_path)
    except Exception as e:
        print(f"Error loading mask for frame {frame_idx}: {e}")
        mask = None

    # Load object analysis
    try:
        with open(analysis_path, 'r') as f:
            analysis = json.load(f)
    except Exception as e:
        print(f"Error loading analysis for frame {frame_idx}: {e}")
        analysis = None

    return frame, depth, mask, analysis

def process_physics_sequence(start_frame, end_frame, base_dir=BASE_DIR, frame_rate=FRAME_RATE):
    """Process a sequence of frames for physics analysis."""
    # Create output directory
    os.makedirs(os.path.join(base_dir, "physics_viz"), exist_ok=True)

    # Track objects across frames
    object_history = {}  # tracking_id -> list of (frame_idx, object_data)
    previous_objects = None
    tracked_objects_by_frame = {}  # Store tracked objects for each frame

    print(f"Processing frames {start_frame} to {end_frame} for object tracking...")

    # First pass: track objects across frames
    for frame_idx in range(start_frame, end_frame + 1):
        # Load frame data
        _, _, _, objects = load_frame_data(frame_idx, base_dir)
        if not objects:
            continue

        # Track objects
        objects = track_objects_between_frames(previous_objects, objects)
        previous_objects = objects

        # Store tracked objects for this frame
        tracked_objects_by_frame[frame_idx] = objects.copy()

        # Update object history
        for obj in objects:
            tracking_id = obj["tracking_id"]
            if tracking_id not in object_history:
                object_history[tracking_id] = []
            object_history[tracking_id].append((frame_idx, obj))

        if frame_idx % 10 == 0:
            print(f"Tracked objects in frame {frame_idx}/{end_frame}")

    # Filter object history to keep only objects with sufficient frames
    filtered_history = {tid: frames for tid, frames in object_history.items() if len(frames) >= 3}

    print(f"Analyzing physics for {len(filtered_history)} tracked objects...")

    # Analyze physics properties
    physics_data = analyze_physics(filtered_history, frame_rate)

    print(f"Creating physics visualizations...")

    # Second pass: create visualizations with better error handling
    viz_count = 0
    for frame_idx in range(start_frame, end_frame + 1):
        try:
            # Load frame data
            frame, depth, mask, _ = load_frame_data(frame_idx, base_dir)

            # Use tracked objects for this frame instead of loading from file again
            objects = tracked_objects_by_frame.get(frame_idx)

            if frame is None or objects is None:
                print(f"Skipping frame {frame_idx} - missing required data")
                continue

            # Analyze game-specific physics
            game_physics = analyze_game_physics(objects, physics_data, depth)

            # Create visualization
            output_path = os.path.join(base_dir, "physics_viz", f"physics_{frame_idx:04d}.png")
            vis_result = create_physics_visualization(frame, objects, physics_data, game_physics, depth, output_path)

            if vis_result is not None:
                viz_count += 1
                if frame_idx % 10 == 0:
                    print(f"Created physics visualization for frame {frame_idx}/{end_frame}")
            else:
                print(f"Failed to create visualization for frame {frame_idx}")
        except Exception as e:
            print(f"Error creating visualization for frame {frame_idx}: {e}")

    print(f"Created {viz_count} physics visualizations")
    return physics_data

def create_physics_video(start_frame, end_frame, output_path, base_dir=BASE_DIR, fps=30):
    """Create a video from the physics visualizations."""
    # Get all visualization files that exist within the range
    physics_viz_dir = os.path.join(base_dir, "physics_viz")
    viz_files = []

    for frame_idx in range(start_frame, end_frame + 1):
        viz_path = os.path.join(physics_viz_dir, f"physics_{frame_idx:04d}.png")
        if os.path.exists(viz_path):
            viz_files.append(viz_path)

    if not viz_files:
        print("No physics visualization files found!")
        return False

    # Get dimensions from first available visualization
    first_frame = cv2.imread(viz_files[0])
    if first_frame is None:
        print(f"Failed to read first visualization frame")
        return False

    height, width = first_frame.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure output directory exists
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Add frames to video
    frames_written = 0
    for viz_path in viz_files:
        frame = cv2.imread(viz_path)
        if frame is not None:
            video.write(frame)
            frames_written += 1
        else:
            print(f"Warning: Could not read frame from {viz_path}")

    # Release video writer
    video.release()

    # Test if the video file was created successfully
    if not os.path.exists(output_path):
        print(f"ERROR: Video file was not created at {output_path}")
        return False

    # Test if the video can be opened and read
    test_cap = cv2.VideoCapture(output_path)
    if not test_cap.isOpened():
        print(f"ERROR: Created video file cannot be opened: {output_path}")
        test_cap.release()
        return False

    # Check video properties
    video_width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = test_cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read a few frames to make sure the video is readable
    frames_read = 0
    for _ in range(min(5, frame_count)):
        ret, frame = test_cap.read()
        if ret:
            frames_read += 1

    test_cap.release()

    # Print diagnostic information
    print(f"Video creation diagnostics:")
    print(f"- Frames written: {frames_written}/{len(viz_files)}")
    print(f"- Output dimensions: {video_width}x{video_height} (expected: {width}x{height})")
    print(f"- FPS: {video_fps} (expected: {fps})")
    print(f"- Frame count: {frame_count} (expected: {frames_written})")
    print(f"- Test frames successfully read: {frames_read}/5")

    if frames_read > 0 and frame_count > 0:
        print(f"SUCCESS: Created and verified physics visualization video with {frame_count} frames at {output_path}")
        # Calculate file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"- File size: {file_size_mb:.2f} MB")
        return True
    else:
        print(f"ERROR: Video file was created but appears to be corrupted or unreadable")
        return False


# === Command Line Interface ===
def main():
    parser = argparse.ArgumentParser(description="Physics Analysis and Visualization")
    parser.add_argument("--base_dir", type=str, default=BASE_DIR, help="Base directory for data")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame index")
    parser.add_argument("--fps", type=int, default=FRAME_RATE, help="Frame rate for physics calculations")
    parser.add_argument("--create_video", action="store_true", help="Create video from visualizations")
    parser.add_argument("--video_output", type=str, default="../data/physics_video.mp4", help="Output video path")
    parser.add_argument("--single_frame", type=int, default=None, help="Process only a single frame")

    args = parser.parse_args()

    # Set up base directory
    base_dir = args.base_dir

    # Determine frame range
    if args.single_frame is not None:
        # Process single frame mode
        start_frame = args.single_frame
        end_frame = args.single_frame
    else:
        # Get all frame files to determine range
        frame_files = sorted([f for f in os.listdir(os.path.join(base_dir, "frames")) if f.endswith(".png")])
        if not frame_files:
            print("No frame files found!")
            return

        # Default frame range if not specified
        if args.start_frame is None:
            args.start_frame = 0

        if args.end_frame is None:
            args.end_frame = int(frame_files[-1].split("_")[1].split(".")[0])

        start_frame = args.start_frame
        end_frame = args.end_frame

    print(f"Processing frames {start_frame} to {end_frame}")

    # Process physics
    physics_data = process_physics_sequence(start_frame, end_frame, base_dir, args.fps)

    # Create video if requested
    if args.create_video:
        create_physics_video(start_frame, end_frame, args.video_output, base_dir, args.fps)

if __name__ == "__main__":
    main()


# === Command Line Interface ===
def main():
    parser = argparse.ArgumentParser(description="Physics Analysis and Visualization")
    parser.add_argument("--base_dir", type=str, default=BASE_DIR, help="Base directory for data")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame index")
    parser.add_argument("--fps", type=int, default=FRAME_RATE, help="Frame rate for physics calculations")
    parser.add_argument("--create_video", action="store_true", help="Create video from visualizations")
    parser.add_argument("--video_output", type=str, default="../data/physics_video.mp4", help="Output video path")
    parser.add_argument("--single_frame", type=int, default=None, help="Process only a single frame")

    args = parser.parse_args()

    # Set up base directory
    base_dir = args.base_dir

    # Determine frame range
    if args.single_frame is not None:
        # Process single frame mode
        start_frame = args.single_frame
        end_frame = args.single_frame
    else:
        # Get all frame files to determine range
        frame_files = sorted([f for f in os.listdir(os.path.join(base_dir, "frames")) if f.endswith(".png")])
        if not frame_files:
            print("No frame files found!")
            return

        # Default frame range if not specified
        if args.start_frame is None:
            args.start_frame = 0

        if args.end_frame is None:
            args.end_frame = int(frame_files[-1].split("_")[1].split(".")[0])

        start_frame = args.start_frame
        end_frame = args.end_frame

    print(f"Processing frames {start_frame} to {end_frame}")

    # Process physics
    physics_data = process_physics_sequence(start_frame, end_frame, base_dir, args.fps)

    # Create video if requested
    if args.create_video:
        create_physics_video(start_frame, end_frame, args.video_output, base_dir, args.fps)

if __name__ == "__main__":
    main()
