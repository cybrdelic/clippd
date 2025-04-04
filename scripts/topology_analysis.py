import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from skimage import measure
from pathlib import Path
from scipy.spatial import distance, Delaunay
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from itertools import combinations
from collections import defaultdict

# Add necessary paths from existing files
sys.path.append("C:/Users/alexf/software-projects/clippd/midas")

# === Configuration ===
BASE_DIR = "../data"
FRAME_DIR = os.path.join(BASE_DIR, "frames")
DEPTH_DIR = os.path.join(BASE_DIR, "depth")
SEGMENTATION_DIR = os.path.join(BASE_DIR, "segmentation_masks")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")
PHYSICS_DIR = os.path.join(BASE_DIR, "physics_viz")
TOPOLOGY_DIR = os.path.join(BASE_DIR, "topology")
FRAME_RATE = 30  # Default frame rate for game footage

# === Helper Functions ===
def load_frame_data(frame_idx, base_dir=BASE_DIR):
    """Load frame, depth, mask, and analysis data for a given frame index."""
    frame_path = os.path.join(base_dir, "frames", f"frame_{frame_idx:04d}.png")
    depth_path = os.path.join(base_dir, "depth", f"depth_{frame_idx:04d}.png")
    mask_path = os.path.join(base_dir, "segmentation_masks", f"mask_{frame_idx:04d}.png")
    analysis_path = os.path.join(base_dir, "analysis", f"analysis_{frame_idx:04d}.json")

    # Check if files exist
    if not all(os.path.exists(p) for p in [frame_path, depth_path, mask_path, analysis_path]):
        print(f"Missing required files for frame {frame_idx}")
        return None, None, None, None

    # Load frame
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Load depth
    depth = plt.imread(depth_path)
    if depth.ndim == 3 and depth.shape[2] == 4:  # RGBA
        depth = depth[:,:,0]  # Get first channel
    elif depth.ndim == 3 and depth.shape[2] == 3:  # RGB
        depth = np.mean(depth, axis=2)  # Convert to grayscale

    # Load mask
    mask = plt.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[:,:,0]  # Get first channel

    # Load object analysis
    with open(analysis_path, 'r') as f:
        analysis = json.load(f)

    return frame, depth, mask, analysis

# === Topological Analysis Functions ===
def build_topology_graph(objects, depth_map):
    """
    Build a topological graph from segmented objects and depth information.

    Args:
        objects (list): List of object dictionaries with centroid, bbox, etc.
        depth_map (np.ndarray): Depth map for the frame

    Returns:
        nx.Graph: Network graph representing scene topology
    """
    # Create empty graph
    G = nx.Graph()

    # Add nodes for each object
    for obj in objects:
        # Extract object properties
        obj_id = obj["id"]
        centroid = obj["centroid"]
        depth = obj["mean_depth"]
        area = obj["area"]

        # Add node with attributes
        G.add_node(obj_id,
                  pos=(centroid[1], centroid[0]),  # x, y format for visualization
                  depth=depth,
                  area=area,
                  bbox=obj["bbox"],
                  relative_position=obj["relative_position"])

    # Generate connections based on different criteria

    # 1. Proximity connections (closer objects are connected)
    centroids = np.array([G.nodes[n]["pos"] for n in G.nodes])
    if len(centroids) > 1:  # Need at least 2 objects
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(centroids)

            # Add edges for triangulation
            for simplex in tri.simplices:
                node_ids = list(G.nodes())
                for i, j in combinations(simplex, 2):
                    G.add_edge(node_ids[i], node_ids[j],
                              relation="proximity",
                              weight=np.linalg.norm(centroids[i] - centroids[j]))
        except Exception as e:
            print(f"Error in Delaunay triangulation: {e}")
            # Fallback: connect nearby objects based on distance threshold
            dist_matrix = squareform(pdist(centroids))
            threshold = np.percentile(dist_matrix[dist_matrix > 0], 25)  # 25th percentile of distances
            for i, j in combinations(range(len(centroids)), 2):
                if dist_matrix[i, j] < threshold:
                    node_ids = list(G.nodes())
                    G.add_edge(node_ids[i], node_ids[j],
                              relation="proximity",
                              weight=dist_matrix[i, j])

    # 2. Depth-based connections (objects at similar depths)
    depths = [G.nodes[n]["depth"] for n in G.nodes]
    if len(depths) > 1:
        depth_diff = np.abs(np.subtract.outer(depths, depths))
        threshold = np.percentile(depth_diff[depth_diff > 0], 15)  # 15th percentile

        node_ids = list(G.nodes())
        for i, j in combinations(range(len(depths)), 2):
            if depth_diff[i, j] < threshold:
                G.add_edge(node_ids[i], node_ids[j],
                          relation="depth_similar",
                          weight=depth_diff[i, j])

    # 3. Occlusion relationships (if we have physics tracking data)
    # This would require object tracking data from previous frames

    return G

def analyze_scene_topology(G, depth_map):
    """
    Analyze scene topology to extract structural information.

    Args:
        G (nx.Graph): Network graph of the scene
        depth_map (np.ndarray): Depth map for the frame

    Returns:
        dict: Topological analysis results
    """
    topology_analysis = {
        "scene_structure": "unknown",
        "foreground_objects": [],
        "background_objects": [],
        "central_objects": [],
        "object_relations": [],
        "clusters": [],
        "occlusions": []
    }

    # Skip analysis if graph is empty
    if len(G) == 0:
        return topology_analysis

    # Identify foreground and background objects
    for node in G.nodes():
        if G.nodes[node]["relative_position"] == "foreground":
            topology_analysis["foreground_objects"].append(node)
        else:
            topology_analysis["background_objects"].append(node)

    # Find central objects by degree or betweenness centrality
    if len(G) > 1:
        centrality = nx.betweenness_centrality(G)
        central_nodes = sorted(centrality, key=centrality.get, reverse=True)

        # Consider top 20% as central (minimum 1)
        num_central = max(1, int(len(central_nodes) * 0.2))
        topology_analysis["central_objects"] = central_nodes[:num_central]

    # Detect clusters using community detection
    if len(G) > 2:
        try:
            clusters = list(nx.community.greedy_modularity_communities(G))
            topology_analysis["clusters"] = [list(cluster) for cluster in clusters]
        except Exception as e:
            print(f"Error in cluster detection: {e}")

    # Analyze object relations
    for edge in G.edges(data=True):
        obj1, obj2, data = edge
        relation_type = data.get("relation", "unknown")

        # Get depth order
        depth1 = G.nodes[obj1]["depth"]
        depth2 = G.nodes[obj2]["depth"]

        if depth1 < depth2:
            front_obj, back_obj = obj1, obj2
        else:
            front_obj, back_obj = obj2, obj1

        # Check for potential occlusion by comparing bounding boxes
        bbox1 = G.nodes[obj1]["bbox"]
        bbox2 = G.nodes[obj2]["bbox"]

        # Simplify bounding boxes for overlap calculation
        box1 = [bbox1[1], bbox1[0], bbox1[3], bbox1[2]]  # [x1, y1, x2, y2]
        box2 = [bbox2[1], bbox2[0], bbox2[3], bbox2[2]]

        # Check overlap
        overlap = not (box1[0] > box2[2] or box1[2] < box2[0] or
                       box1[1] > box2[3] or box1[3] < box2[1])

        if overlap and abs(depth1 - depth2) > 0.1:
            # Potential occlusion
            topology_analysis["occlusions"].append({
                "occluder": front_obj,
                "occluded": back_obj,
                "depth_diff": abs(depth1 - depth2)
            })

        # Record relationship
        topology_analysis["object_relations"].append({
            "obj1": obj1,
            "obj2": obj2,
            "relation": relation_type,
            "depth_order": [front_obj, back_obj]
        })

    # Determine overall scene structure
    if len(topology_analysis["clusters"]) > 1:
        topology_analysis["scene_structure"] = "multi_cluster"
    elif len(topology_analysis["central_objects"]) == 1 and len(G) > 3:
        topology_analysis["scene_structure"] = "central_focused"
    elif len(topology_analysis["foreground_objects"]) > 0 and len(topology_analysis["background_objects"]) > 0:
        topology_analysis["scene_structure"] = "foreground_background"
    else:
        topology_analysis["scene_structure"] = "simple"

    return topology_analysis

def extract_persistent_topology(frames_data):
    """
    Analyze topological features that persist across multiple frames.

    Args:
        frames_data (list): List of (graph, topology_analysis) pairs

    Returns:
        dict: Analysis of persistent topological features
    """
    if not frames_data or len(frames_data) < 2:
        return {"persistent_features": []}

    persistent = {
        "stable_clusters": [],
        "stable_relations": [],
        "persistent_central_objects": [],
        "changing_relations": [],
    }

    # Track objects that appear in multiple frames
    object_appearances = defaultdict(list)
    frame_graphs = []

    # Extract graphs and count object appearances
    for frame_idx, (G, analysis) in enumerate(frames_data):
        frame_graphs.append(G)
        for node in G.nodes():
            object_appearances[node].append(frame_idx)

    # Find objects that appear in multiple frames (at least half)
    min_appearances = max(2, len(frames_data) // 2)
    persistent_objects = {obj: frames for obj, frames in object_appearances.items()
                          if len(frames) >= min_appearances}

    # Find persistent central objects
    central_counts = defaultdict(int)
    for _, analysis in frames_data:
        for obj in analysis["central_objects"]:
            central_counts[obj] += 1

    # Objects that are central in at least half their appearances
    for obj, count in central_counts.items():
        if obj in persistent_objects and count >= len(persistent_objects[obj]) // 2:
            persistent["persistent_central_objects"].append(obj)

    # Analyze persistent relationships
    edge_counts = defaultdict(int)
    edge_relations = defaultdict(list)

    for G in frame_graphs:
        for u, v in G.edges():
            if u in persistent_objects and v in persistent_objects:
                edge = tuple(sorted([u, v]))
                edge_counts[edge] += 1
                rel = G.edges[u, v].get("relation", "unknown")
                edge_relations[edge].append(rel)

    # Find stable relationships (edges that appear in most frames)
    for edge, count in edge_counts.items():
        u, v = edge
        # At least present in half of the frames where both objects appear
        possible_frames = set(persistent_objects[u]).intersection(set(persistent_objects[v]))
        if count >= len(possible_frames) // 2:
            # Check if relation is consistent
            relations = edge_relations[edge]
            most_common_relation = max(set(relations), key=relations.count)

            if relations.count(most_common_relation) >= len(relations) * 0.7:
                # Stable relation
                persistent["stable_relations"].append({
                    "objects": list(edge),
                    "relation": most_common_relation,
                    "stability": count / len(possible_frames)
                })
            else:
                # Changing relation
                persistent["changing_relations"].append({
                    "objects": list(edge),
                    "relations": list(set(relations)),
                    "stability": count / len(possible_frames)
                })

    # Identify stable clusters through time
    cluster_history = []
    for _, analysis in frames_data:
        cluster_history.append(analysis["clusters"])

    # Only analyze if we have enough frames with clusters
    if len([c for c in cluster_history if c]) >= len(frames_data) // 2:
        # Find persistent clusters using a simplified approach
        object_cluster_counts = defaultdict(lambda: defaultdict(int))

        for frame_clusters in cluster_history:
            for i, cluster in enumerate(frame_clusters):
                for obj in cluster:
                    if obj in persistent_objects:
                        object_cluster_counts[obj][i] += 1

        # Group objects that frequently appear in the same cluster
        object_groups = defaultdict(set)
        processed = set()

        for obj1, obj2 in combinations(persistent_objects.keys(), 2):
            if obj1 in processed and obj2 in processed:
                continue

            # Check if objects frequently appear in the same cluster
            same_cluster_count = 0
            total_co_appearances = 0

            for frame_idx in set(persistent_objects[obj1]).intersection(set(persistent_objects[obj2])):
                if frame_idx < len(cluster_history) and cluster_history[frame_idx]:
                    total_co_appearances += 1

                    # Check if in same cluster
                    obj1_cluster = None
                    obj2_cluster = None

                    for i, cluster in enumerate(cluster_history[frame_idx]):
                        if obj1 in cluster:
                            obj1_cluster = i
                        if obj2 in cluster:
                            obj2_cluster = i

                    if obj1_cluster is not None and obj1_cluster == obj2_cluster:
                        same_cluster_count += 1

            if total_co_appearances > 0 and same_cluster_count >= total_co_appearances * 0.7:
                # These objects frequently appear in the same cluster
                group_found = False

                for group_id, group in object_groups.items():
                    if obj1 in group or obj2 in group:
                        group.add(obj1)
                        group.add(obj2)
                        group_found = True
                        break

                if not group_found:
                    group_id = len(object_groups)
                    object_groups[group_id] = {obj1, obj2}

                processed.add(obj1)
                processed.add(obj2)

        # Convert to list of stable clusters
        persistent["stable_clusters"] = [list(cluster) for cluster in object_groups.values()]

    return persistent

def compute_structural_importance(G, analysis):
    """
    Compute structural importance scores for each object.

    Args:
        G (nx.Graph): Topology graph
        analysis (dict): Topology analysis results

    Returns:
        dict: Object ID to importance score mapping
    """
    if not G or len(G) == 0:
        return {}

    importance_scores = {}

    # Compute various centrality measures
    degree_cent = nx.degree_centrality(G)

    if len(G) > 1:
        between_cent = nx.betweenness_centrality(G)
        close_cent = nx.closeness_centrality(G)
    else:
        between_cent = {node: 0 for node in G.nodes()}
        close_cent = {node: 0 for node in G.nodes()}

    # Calculate importance as weighted combination of factors
    for node in G.nodes():
        # Base score from centrality metrics (normalized)
        centrality_score = (0.4 * degree_cent.get(node, 0) +
                          0.4 * between_cent.get(node, 0) +
                          0.2 * close_cent.get(node, 0))

        # Boost for foreground objects
        foreground_boost = 1.5 if node in analysis["foreground_objects"] else 1.0

        # Boost for central objects
        central_boost = 1.8 if node in analysis["central_objects"] else 1.0

        # Boost for large objects (based on normalized area)
        area = G.nodes[node]["area"]
        max_area = max([G.nodes[n]["area"] for n in G.nodes()])
        size_boost = 0.5 + 1.0 * (area / max_area)

        # Boost for objects involved in occlusions
        occlusion_boost = 1.0
        for occlusion in analysis["occlusions"]:
            if node == occlusion["occluder"]:
                occlusion_boost = 1.3  # Occluders are important
                break

        # Compute final importance score
        importance = centrality_score * foreground_boost * central_boost * size_boost * occlusion_boost
        importance_scores[node] = importance

    # Normalize scores to [0, 1] range
    if importance_scores:
        max_score = max(importance_scores.values())
        if max_score > 0:
            importance_scores = {k: v/max_score for k, v in importance_scores.items()}

    return importance_scores

def visualize_topology(frame, G, analysis, importance_scores=None, output_path=None):
    """
    Create visualization of scene topology analysis.

    Args:
        frame (np.ndarray): Original video frame
        G (nx.Graph): Topology graph
        analysis (dict): Topology analysis results
        importance_scores (dict, optional): Object importance scores
        output_path (str, optional): Path to save visualization

    Returns:
        np.ndarray: Visualization image
    """
    # Create a copy for visualization
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Overlay for graph visualization
    overlay = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA

    # Define node positions for drawing
    pos = {}
    for node in G.nodes():
        pos[node] = G.nodes[node]["pos"]  # x, y format from graph

    # Draw edges (connections between objects)
    for u, v, data in G.edges(data=True):
        # Get positions
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        # Determine color based on relation type
        relation = data.get("relation", "unknown")

        if relation == "proximity":
            color = (0, 255, 0, 128)  # Green, semi-transparent
        elif relation == "depth_similar":
            color = (255, 0, 0, 128)  # Red, semi-transparent
        else:
            color = (0, 0, 255, 128)  # Blue, semi-transparent

        # Draw line
        cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # Draw nodes (objects)
    for node in G.nodes():
        x, y = pos[node]
        x, y = int(x), int(y)

        # Determine size and color based on importance
        importance = importance_scores.get(node, 0.5) if importance_scores else 0.5

        # Size proportional to importance
        radius = int(10 + 20 * importance)

        # Color based on object type/importance
        if node in analysis["central_objects"]:
            color = (255, 255, 0, 200)  # Yellow for central objects
        elif node in analysis["foreground_objects"]:
            color = (0, 255, 255, 200)  # Cyan for foreground
        else:
            color = (128, 128, 128, 200)  # Gray for background

        # Draw node
        cv2.circle(overlay, (x, y), radius, color, -1)

        # Add node label
        cv2.putText(
            overlay, str(node), (x - 10, y - radius - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255, 255), 1
        )

    # Blend overlay with original frame
    vis_with_overlay = vis.copy()
    alpha_mask = overlay[:,:,3] / 255.0
    for c in range(3):  # RGB channels
        vis_with_overlay[:,:,c] = (1 - alpha_mask) * vis[:,:,c] + alpha_mask * overlay[:,:,c]

    # Add topology information text
    text_y = 30
    text_spacing = 25

    # Title
    cv2.putText(
        vis_with_overlay, "Scene Topology Analysis", (20, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    text_y += text_spacing

    # Scene structure
    cv2.putText(
        vis_with_overlay, f"Structure: {analysis['scene_structure']}", (20, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
    )
    text_y += text_spacing

    # Object counts
    cv2.putText(
        vis_with_overlay, f"Objects: {len(G.nodes())} total, {len(analysis['foreground_objects'])} foreground",
        (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
    )
    text_y += text_spacing

    # Clusters
    cv2.putText(
        vis_with_overlay, f"Clusters: {len(analysis['clusters'])}", (20, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
    )
    text_y += text_spacing

    # Important objects
    if analysis["central_objects"]:
        cv2.putText(
            vis_with_overlay, f"Central objects: {', '.join(map(str, analysis['central_objects']))}",
            (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        text_y += text_spacing

    # Save visualization if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(vis_with_overlay, cv2.COLOR_RGB2BGR))

    return vis_with_overlay

def process_frame_topology(frame_idx, base_dir=BASE_DIR):
    """
    Process scene topology for a single frame.

    Args:
        frame_idx (int): Frame index
        base_dir (str): Base directory for data

    Returns:
        tuple: (G, topology_analysis, visualization)
    """
    # Load frame data
    frame, depth, mask, objects = load_frame_data(frame_idx, base_dir)

    if frame is None or depth is None or objects is None:
        print(f"Could not load necessary data for frame {frame_idx}")
        return None, None, None

    # Create graph from objects
    G = build_topology_graph(objects, depth)

    # Analyze scene topology
    topology_analysis = analyze_scene_topology(G, depth)

    # Compute object importance
    importance_scores = compute_structural_importance(G, topology_analysis)

    # Create visualization
    output_path = os.path.join(base_dir, "topology", f"topology_{frame_idx:04d}.png")
    visualization = visualize_topology(frame, G, topology_analysis, importance_scores, output_path)

    return G, topology_analysis, visualization

def process_sequence_topology(start_frame, end_frame, base_dir=BASE_DIR):
    """
    Process scene topology for a sequence of frames and analyze persistent features.

    Args:
        start_frame (int): Start frame index
        end_frame (int): End frame index
        base_dir (str): Base directory for data

    Returns:
        dict: Persistent topology analysis
    """
    # Create output directory
    os.makedirs(os.path.join(base_dir, "topology"), exist_ok=True)

    # Process frames
    frames_data = []

    for frame_idx in range(start_frame, end_frame + 1):
        print(f"Processing topology for frame {frame_idx}...")

        G, analysis, _ = process_frame_topology(frame_idx, base_dir)

        if G is not None and analysis is not None:
            frames_data.append((G, analysis))

        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx - start_frame + 1}/{end_frame - start_frame + 1} frames")

    # Analyze persistent topological features
    persistent_features = extract_persistent_topology(frames_data)

    # Save persistent analysis
    os.makedirs(os.path.join(base_dir, "analysis"), exist_ok=True)
    persistent_path = os.path.join(base_dir, "analysis", "persistent_topology.json")

    try:
        with open(persistent_path, 'w') as f:
            # Ensure serializable
            serializable_features = {}
            for key, value in persistent_features.items():
                if isinstance(value, list):
                    # Convert any tuples to lists
                    processed_value = []
                    for item in value:
                        if isinstance(item, dict):
                            processed_item = {}
                            for k, v in item.items():
                                if isinstance(v, (tuple, set)):
                                    processed_item[k] = list(v)
                                else:
                                    processed_item[k] = v
                            processed_value.append(processed_item)
                        else:
                            processed_value.append(item)
                    serializable_features[key] = processed_value
                else:
                    serializable_features[key] = value

            json.dump(serializable_features, f, indent=2)
    except Exception as e:
        print(f"Error saving persistent topology analysis: {e}")

    return persistent_features

def create_scene_understanding_visualization(frame_idx, base_dir=BASE_DIR):
    """
    Create a comprehensive visualization that combines depth, object analysis,
    physics, and topology for complete scene understanding.

    Args:
        frame_idx (int): Frame index
        base_dir (str): Base directory for data

    Returns:
        np.ndarray: Visualization image
    """
    # Load all available data for this frame
    frame, depth, mask, objects = load_frame_data(frame_idx, base_dir)

    if frame is None:
        print(f"Missing frame data for {frame_idx}")
        return None

    # Create base visualization with original frame
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Create multi-layer visualization

    # Layer 1: Blend with depth map
    if depth is not None:
        # Convert depth to colormap
        if depth.ndim == 2:
            depth_colored = plt.cm.plasma(plt.Normalize()(depth))[:,:,:3]
        else:
            depth_colored = depth

        # Blend with original frame
        vis_with_depth = vis.copy().astype(float) / 255.0
        for c in range(3):
            vis_with_depth[:,:,c] = vis_with_depth[:,:,c] * 0.7 + depth_colored[:,:,c] * 0.3

        vis = (vis_with_depth * 255).astype(np.uint8)

    # Layer 2: Add object segmentation and physics
    if mask is not None and objects is not None:
        # Create a semi-transparent overlay
        overlay = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA

        # Draw object boundaries
        for obj in objects:
            # Get bounding box
            y1, x1, y2, x2 = [int(coord) for coord in obj["bbox"]]

            # Determine color based on relative position
            if obj["relative_position"] == "foreground":
                color = (255, 50, 50, 180)  # Red for foreground
            else:
                color = (50, 50, 255, 180)  # Blue for background

            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Add object ID
            cv2.putText(
                overlay, f"ID:{obj['id']}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255, 200), 1
            )

        # Blend overlay with frame
        alpha_mask = overlay[:,:,3] / 255.0
        for c in range(3):
            vis[:,:,c] = (1 - alpha_mask) * vis[:,:,c] + alpha_mask * overlay[:,:,c]

    # Layer 3: Add topology analysis
    try:
        # Process topology if not already done
        G, topology_analysis, _ = process_frame_topology(frame_idx, base_dir)

        if G is not None and topology_analysis is not None:
            # Add topology overlay
            topology_overlay = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA

            # Draw graph edges (connections between objects)
            for u, v, data in G.edges(data=True):
                # Get node positions
                x1, y1 = G.nodes[u]["pos"]
                x2, y2 = G.nodes[v]["pos"]

                # Determine relationship type
                relation = data.get("relation", "unknown")

                if relation == "proximity":
                    color = (0, 255, 0, 100)  # Green, light opacity
                elif relation == "depth_similar":
                    color = (255, 255, 0, 100)  # Yellow, light opacity
                else:
                    color = (0, 255, 255, 100)  # Cyan, light opacity

                # Draw line with arrow indicating relationship
                cv2.arrowedLine(
                    topology_overlay,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color, 1, tipLength=0.1
                )

            # Add semantic labels for important clusters and scene structure
            if topology_analysis["scene_structure"] != "unknown":
                # Add scene structure label
                structure_label = f"Structure: {topology_analysis['scene_structure']}"
                cv2.putText(
                    topology_overlay, structure_label, (w - 300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255, 255), 1
                )

            # Add labels for object clusters
            for i, cluster in enumerate(topology_analysis["clusters"]):
                if not cluster:
                    continue

                # Find average position of cluster
                coords = []
                for obj_id in cluster:
                    for obj in objects:
                        if obj["id"] == obj_id:
                            coords.append((obj["centroid"][1], obj["centroid"][0]))  # x, y format
                            break

                if coords:
                    center_x = int(sum(c[0] for c in coords) / len(coords))
                    center_y = int(sum(c[1] for c in coords) / len(coords))

                    # Draw cluster label
                    cv2.putText(
                        topology_overlay, f"Cluster {i+1}", (center_x, center_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0, 255), 1
                    )

            # Blend topology overlay with visualization
            alpha_mask = topology_overlay[:,:,3] / 255.0
            for c in range(3):
                vis[:,:,c] = (1 - alpha_mask) * vis[:,:,c] + alpha_mask * topology_overlay[:,:,c]
    except Exception as e:
        print(f"Error adding topology layer: {e}")

    # Add informative text about the scene
    text_y = 30
    text_spacing = 25

    # Add title
    cv2.putText(
        vis, "Scene Understanding", (20, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    text_y += text_spacing

    # Add frame info
    cv2.putText(
        vis, f"Frame: {frame_idx}", (20, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
    )
    text_y += text_spacing

    # Add object count
    if objects:
        cv2.putText(
            vis, f"Objects: {len(objects)}", (20, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        text_y += text_spacing

    # Add topology info if available
    if 'topology_analysis' in locals() and topology_analysis:
        # Add central objects
        if topology_analysis["central_objects"]:
            central_str = f"Central: {', '.join(map(str, topology_analysis['central_objects']))}"
            cv2.putText(
                vis, central_str, (20, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )
            text_y += text_spacing

        # Add occlusion info
        if topology_analysis["occlusions"]:
            occlusion_str = f"Occlusions: {len(topology_analysis['occlusions'])}"
            cv2.putText(
                vis, occlusion_str, (20, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )
            text_y += text_spacing

    # Save the combined visualization
    output_path = os.path.join(base_dir, "topology", f"scene_understanding_{frame_idx:04d}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    return vis

def integrate_tracking_with_topology(frame_idx, base_dir=BASE_DIR):
    """
    Integrate object tracking information with topology to enable temporal
    reasoning about the scene structure.

    Args:
        frame_idx (int): Current frame index
        base_dir (str): Base directory for data

    Returns:
        dict: Enhanced scene understanding with temporal context
    """
    # Load tracking data for surrounding frames (if available)
    temporal_context = {
        "tracking": {},
        "topology_evolution": {},
        "scene_events": []
    }

    # Check for tracking data
    tracking_file = os.path.join(base_dir, "physics_viz", f"physics_{frame_idx:04d}.png")
    if not os.path.exists(tracking_file):
        # If no tracking data, we can't do temporal analysis
        return temporal_context

    # Get current frame topology
    _, curr_topology, _ = process_frame_topology(frame_idx, base_dir)
    if not curr_topology:
        return temporal_context

    # Try to get previous frame topology (if exists)
    prev_topology = None
    prev_file = os.path.join(base_dir, "topology", f"topology_{frame_idx-1:04d}.png")
    if os.path.exists(prev_file):
        _, prev_topology, _ = process_frame_topology(frame_idx-1, base_dir)

    # Detect changes in topology between frames
    if prev_topology:
        # Track changes in central objects
        curr_central = set(curr_topology["central_objects"])
        prev_central = set(prev_topology["central_objects"])

        new_central = curr_central - prev_central
        lost_central = prev_central - curr_central

        if new_central:
            temporal_context["scene_events"].append({
                "event_type": "new_central_objects",
                "objects": list(new_central),
                "frame": frame_idx
            })

        if lost_central:
            temporal_context["scene_events"].append({
                "event_type": "lost_central_objects",
                "objects": list(lost_central),
                "frame": frame_idx
            })

        # Track changes in clusters
        if curr_topology["clusters"] and prev_topology["clusters"]:
            # Simple heuristic to identify new or dissolved clusters
            curr_cluster_sizes = [len(c) for c in curr_topology["clusters"]]
            prev_cluster_sizes = [len(c) for c in prev_topology["clusters"]]

            if len(curr_topology["clusters"]) > len(prev_topology["clusters"]):
                temporal_context["scene_events"].append({
                    "event_type": "cluster_formation",
                    "frame": frame_idx
                })
            elif len(curr_topology["clusters"]) < len(prev_topology["clusters"]):
                temporal_context["scene_events"].append({
                    "event_type": "cluster_dissolution",
                    "frame": frame_idx
                })

        # Track changes in occlusions
        curr_occlusions = {(o["occluder"], o["occluded"]) for o in curr_topology["occlusions"]}
        prev_occlusions = {(o["occluder"], o["occluded"]) for o in prev_topology["occlusions"]}

        new_occlusions = curr_occlusions - prev_occlusions
        ended_occlusions = prev_occlusions - curr_occlusions

        if new_occlusions:
            temporal_context["scene_events"].append({
                "event_type": "new_occlusions",
                "occlusions": [{"occluder": a, "occluded": b} for a, b in new_occlusions],
                "frame": frame_idx
            })

        if ended_occlusions:
            temporal_context["scene_events"].append({
                "event_type": "ended_occlusions",
                "occlusions": [{"occluder": a, "occluded": b} for a, b in ended_occlusions],
                "frame": frame_idx
            })

        # Track overall scene structure changes
        if curr_topology["scene_structure"] != prev_topology["scene_structure"]:
            temporal_context["scene_events"].append({
                "event_type": "structure_change",
                "from": prev_topology["scene_structure"],
                "to": curr_topology["scene_structure"],
                "frame": frame_idx
            })

    # Store topology evolution data
    temporal_context["topology_evolution"] = {
        "current": curr_topology,
        "previous": prev_topology
    }

    return temporal_context

def enhanced_3d_scene_understanding(frame_idx, base_dir=BASE_DIR):
    """
    Combine depth, object tracking, and topology to create an enhanced 3D
    understanding of the scene.

    Args:
        frame_idx (int): Frame index
        base_dir (str): Base directory for data

    Returns:
        dict: Enhanced 3D scene understanding
    """
    # Load all data sources
    frame, depth, mask, objects = load_frame_data(frame_idx, base_dir)

    if frame is None or depth is None or objects is None:
        print(f"Missing required data for frame {frame_idx}")
        return None

    # Process topology
    G, topology_analysis, _ = process_frame_topology(frame_idx, base_dir)

    # Integrate tracking and temporal context
    temporal_context = integrate_tracking_with_topology(frame_idx, base_dir)

    # Create enhanced 3D scene understanding
    scene_3d = {
        "frame_idx": frame_idx,
        "timestamp": frame_idx / FRAME_RATE,  # Convert to seconds
        "objects_3d": [],
        "spatial_relations": [],
        "scene_structure": topology_analysis["scene_structure"] if topology_analysis else "unknown",
        "temporal_context": temporal_context
    }

    # Enhanced object properties with depth and topology
    for obj in objects:
        obj_id = obj["id"]

        # Get centroid and depth
        y, x = obj["centroid"]
        depth_value = obj["mean_depth"]

        # Calculate 3D position (using a simple projection)
        # We're using a simplified pinhole camera model here
        # In a real system, you'd use camera intrinsics/extrinsics
        # This is just an estimation for visualization

        # Assume center of image is optical axis
        h, w = frame.shape[:2]
        cx, cy = w/2, h/2

        # Simple focal length estimation (depends on camera)
        # This is just a placeholder - would need calibration in real use
        focal_length = w  # Reasonable default estimate

        # Calculate approximate 3D coordinates
        # Z is depth
        z = depth_value
        # X and Y are proportional to image coordinates and depth
        x_3d = (x - cx) * z / focal_length
        y_3d = (y - cy) * z / focal_length

        # Create enhanced 3D object representation
        obj_3d = {
            "id": obj_id,
            "image_pos": (float(x), float(y)),
            "depth": float(depth_value),
            "estimated_3d_pos": (float(x_3d), float(y_3d), float(z)),
            "size": float(obj["area"]),
            "bounding_box": [float(coord) for coord in obj["bbox"]]
        }

        # Add topology information if available
        if G and obj_id in G:
            # Get importance score
            importance = compute_structural_importance(G, topology_analysis).get(obj_id, 0)
            obj_3d["importance"] = float(importance)

            # Get topological role
            if obj_id in topology_analysis["central_objects"]:
                obj_3d["topo_role"] = "central"
            elif obj_id in topology_analysis["foreground_objects"]:
                obj_3d["topo_role"] = "foreground"
            else:
                obj_3d["topo_role"] = "background"

        scene_3d["objects_3d"].append(obj_3d)

    # Extract spatial relations between objects
    if G:
        for u, v, data in G.edges(data=True):
            relation = data.get("relation", "unknown")

            # Get 3D positions
            obj_u = next((o for o in scene_3d["objects_3d"] if o["id"] == u), None)
            obj_v = next((o for o in scene_3d["objects_3d"] if o["id"] == v), None)

            if obj_u and obj_v:
                # Calculate 3D distance
                pos_u = obj_u["estimated_3d_pos"]
                pos_v = obj_v["estimated_3d_pos"]
                dist_3d = np.sqrt(sum((a-b)**2 for a, b in zip(pos_u, pos_v)))

                # Determine depth relation
                if obj_u["depth"] < obj_v["depth"]:
                    depth_relation = "in_front_of"
                else:
                    depth_relation = "behind"

                # Add spatial relation
                scene_3d["spatial_relations"].append({
                    "object1": u,
                    "object2": v,
                    "relation_type": relation,
                    "depth_relation": depth_relation,
                    "distance_3d": float(dist_3d)
                })

    # Save the enhanced 3D understanding
    os.makedirs(os.path.join(base_dir, "analysis"), exist_ok=True)
    scene_3d_path = os.path.join(base_dir, "analysis", f"scene3d_{frame_idx:04d}.json")

    with open(scene_3d_path, 'w') as f:
        json.dump(scene_3d, f, indent=2)

    return scene_3d

def create_3d_visualization(scene_3d, output_path=None):
    """
    Create a 3D visualization of the scene understanding.

    Args:
        scene_3d (dict): Enhanced 3D scene understanding
        output_path (str, optional): Path to save visualization

    Returns:
        np.ndarray: Visualization image
    """
    # Use matplotlib for 3D visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Extract 3D positions for all objects
    positions = []
    colors = []
    sizes = []
    labels = []

    for obj in scene_3d["objects_3d"]:
        pos = obj["estimated_3d_pos"]
        positions.append(pos)

        # Determine color based on topological role
        if "topo_role" in obj:
            if obj["topo_role"] == "central":
                color = 'red'
            elif obj["topo_role"] == "foreground":
                color = 'green'
            else:
                color = 'blue'
        else:
            color = 'gray'

        colors.append(color)

        # Size based on importance or original size
        if "importance" in obj:
            size = 100 + 900 * obj["importance"]  # Scale for visualization
        else:
            # Use sqrt of area for reasonable marker size
            size = np.sqrt(obj["size"]) / 5

        sizes.append(size)
        labels.append(str(obj["id"]))

    # Convert to numpy arrays
    positions = np.array(positions)

    # Plot 3D points representing objects
    if len(positions) > 0:
        x, y, z = positions[:,0], positions[:,1], positions[:,2]
        scatter = ax.scatter(x, y, z, c=colors, s=sizes, alpha=0.8)

        # Add labels to points
        for i, (x, y, z, label) in enumerate(zip(x, y, z, labels)):
            ax.text(x, y, z, label, fontsize=9)

        # Add spatial relations as lines
        for relation in scene_3d["spatial_relations"]:
            obj1_id = relation["object1"]
            obj2_id = relation["object2"]

            # Find positions
            obj1 = next((o for o in scene_3d["objects_3d"] if o["id"] == obj1_id), None)
            obj2 = next((o for o in scene_3d["objects_3d"] if o["id"] == obj2_id), None)

            if obj1 and obj2:
                pos1 = obj1["estimated_3d_pos"]
                pos2 = obj2["estimated_3d_pos"]

                # Determine line style based on relation type
                if relation["relation_type"] == "proximity":
                    linestyle = '-'
                    color = 'gray'
                elif relation["relation_type"] == "depth_similar":
                    linestyle = '--'
                    color = 'green'
                else:
                    linestyle = ':'
                    color = 'blue'

                # Draw line
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                       linestyle=linestyle, color=color, alpha=0.5)

    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Depth)')
    ax.set_title(f'3D Scene Understanding (Frame {scene_3d["frame_idx"]})')

    # Add scene structure info
    structure_text = f"Structure: {scene_3d['scene_structure']}"
    fig.text(0.05, 0.95, structure_text, fontsize=10)

    # Add temporal event info if available
    if scene_3d["temporal_context"]["scene_events"]:
        events_text = "Events: " + ", ".join(e["event_type"] for e in scene_3d["temporal_context"]["scene_events"])
        fig.text(0.05, 0.92, events_text, fontsize=9)

    # Save visualization if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)

    # Create a numpy array of the visualization by rendering to buffer
    fig.canvas.draw()
    vis = np.array(fig.canvas.renderer.buffer_rgba())

    # Close figure to prevent display in notebooks
    plt.close(fig)

    return vis

# === Main Script ===
def main():
    parser = argparse.ArgumentParser(description="Topological Scene Analysis")
    parser.add_argument("--base_dir", type=str, default=BASE_DIR, help="Base directory for data")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame index")
    parser.add_argument("--single_frame", type=int, default=None, help="Process a single frame")
    parser.add_argument("--create_3d", action="store_true", help="Create 3D visualizations")

    args = parser.parse_args()

    # Set up directories
    base_dir = args.base_dir
    os.makedirs(os.path.join(base_dir, "topology"), exist_ok=True)

    # Determine frame range
    if args.single_frame is not None:
        # Process a single frame
        frames = [args.single_frame]
    else:
        # Find all frame files
        frame_files = sorted([f for f in os.listdir(os.path.join(base_dir, "frames")) if f.endswith(".png")])

        if not frame_files:
            print("No frame files found!")
            return

        # Extract frame indices
        frame_indices = [int(f.split("_")[1].split(".")[0]) for f in frame_files]

        # Determine start and end frames
        start_frame = args.start_frame if args.start_frame is not None else min(frame_indices)
        end_frame = args.end_frame if args.end_frame is not None else max(frame_indices)

        frames = list(range(start_frame, end_frame + 1))

    # Process frames
    for frame_idx in frames:
        print(f"Processing topological analysis for frame {frame_idx}...")

        # Process basic topology
        G, topology_analysis, _ = process_frame_topology(frame_idx, base_dir)

        if G is None or topology_analysis is None:
            print(f"Failed to process topology for frame {frame_idx}")
            continue

        # Create comprehensive scene understanding visualization
        scene_vis = create_scene_understanding_visualization(frame_idx, base_dir)

        # Perform enhanced 3D scene understanding
        scene_3d = enhanced_3d_scene_understanding(frame_idx, base_dir)

        # Create 3D visualization if requested
        if args.create_3d and scene_3d is not None:
            output_path = os.path.join(base_dir, "topology", f"3d_scene_{frame_idx:04d}.png")
            create_3d_visualization(scene_3d, output_path)

        print(f"Completed processing for frame {frame_idx}")

    # If multiple frames were processed, analyze persistent topology
    if len(frames) > 1:
        print("Analyzing persistent topology across frames...")
        persistent_features = process_sequence_topology(min(frames), max(frames), base_dir)
        print(f"Found {len(persistent_features['stable_clusters'])} stable clusters and "
              f"{len(persistent_features['persistent_central_objects'])} persistent central objects")

    print("Topological analysis complete!")

if __name__ == "__main__":
    main()
