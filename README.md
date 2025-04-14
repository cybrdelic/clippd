

# Clippd: Cinematic Scene Understanding Pipeline

_A kickass compendium of advanced computer vision, 3D scene understanding, physics, and topology analysis rolled into one crazy pipeline. Designed for those who want to automate, analyze, and visualize cinematic content, game footage, or any 3D scene with high-tech flair._

---

## Overview

This project is a multi-stage pipeline engineered for **dense 3D scene analysis**. It integrates:

- **Depth Estimation** using **MiDaS** (with robust scale-shift fusion, RANSAC correction, and multi-layer segmentation) to produce precise depth maps.
- **Instance & Semantic Segmentation** that blends Mask R-CNN with SAM 2 for improved segmentation—ranging from global scene breakdowns to fine-grained object isolation.
- **Physics Analysis & Tracking** which computes real-time object trajectories, velocities, accelerations, and even predicts future positions using projectile and linear motion models.
- **Topological & 3D Scene Analysis** that builds a spatial graph (via Delaunay triangulation and centrality measures), detects clusters and occlusions, and fuses it with a 3D reconstruction / SLAM module.
- **User Interface (GUI)** built on PyQt5 to provide a convenient way to run various pipeline steps, preview results, and export final outputs—perfect for when you need visuals, rn.

This project is *not* for the faint-hearted; it’s designed for ambitious doers who enjoy tech at +2sd smarter levels and don’t shy away from complex automation and deep learning wizardry.

---

## Features

- **Frame Extraction** from input video clips with customizable extraction.
- **MiDaS-Powered Depth Estimation:** Convert frames into accurate depth maps while preserving colormaps for visualization.
- **Enhanced Segmentation:**
  - **Instance segmentation** (Mask R-CNN) meets **SAM 2** to yield refined object masks.
  - **Multi-scale and hierarchical segmentation** to capture objects at various depths and scales.
- **Physics & Trajectory Analysis:**
  - Tracking of objects across frames (using centroid, area, and depth similarity).
  - Velocity/acceleration calculations and projectile vs. linear motion predictions.
  - Generation of physics overlays and even full-fledged physics visualization videos.
- **Topological Scene Analysis:**
  - Construction of scene graphs with nodes representing objects and edges based on spatial proximity and depth relations.
  - Community detection to identify clusters, persistent central objects, and occlusion relationships.
- **3D Reconstruction / SLAM Integration:**
  - ORB feature matching and triangulation create a 3D point cloud.
  - 3D terrain wireframe mapping using matplotlib’s 3D plotting.
- **PyQt5 GUI:** Provides a slick interface to configure pipeline parameters, run different pipeline stages, and preview/export results.

---

## Prerequisites

Before you run this beast of a pipeline, ensure you have these dependencies installed:

- **Python 3.8+**
- **PyTorch** (with CUDA support if you have a compatible GPU)
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **SciPy**
- **scikit-image**
- **torchvision**
- **PyQt5**
- **Hydra-core**
- **tqdm** (for progress bars, optional but recommended)
- **networkx**
- **Pillow**
- **numpy-stl** (for STL export in mesh generation)
- **SAM 2** dependencies (make sure to set the correct file paths for checkpoint and config)

*Tip:* Some modules (e.g., cv2.ximgproc) are optional. If missing, the pipeline falls back gracefully.

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/cinematic-pipeline.git
   cd cinematic-pipeline
   ```

2. **Set Up a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   *Note:* Your `requirements.txt` should list all necessary packages. If certain packages aren’t available, install them individually (e.g., `pip install opencv-python-headless torch torchvision PyQt5 hydra-core networkx`).

4. **Download Models and Weights:**

   - **MiDaS Model Weights:**
     Place the weights (e.g., `dpt_large_384.pt`) in the `weights/` directory.
   - **SAM 2 Checkpoint & Config:**
     Download the SAM 2 checkpoint (`sam2.1_hiera_large.pt`) and ensure the config file (`sam2.1_hiera_l.yaml`) is placed at the correct location as per the code (or adjust the paths).

---

## Project Structure

The project is organized into several modules, each handling a component of the pipeline:

- `depth_estimation.py`
  Implements MiDaS-based depth prediction and saving depth visualizations.

- `segmentation.py`
  Contains functions for both instance segmentation (Mask R-CNN) and SAM 2 segmentation, including multi-scale and layered segmentation.

- `physics_analysis.py`
  Covers tracking between frames, physics parameter computation (velocity, acceleration, projectile fits), and predicts future object positions. Also includes utilities to create physics visualizations and videos using `create_physics_video`.

- `topology_analysis.py`
  Constructs a scene graph from segmented objects, runs Delaunay triangulation, computes centrality, occlusion, and cluster information, and visualizes the topology.

- `slam_and_3d_reconstruction.py`
  Implements basic SLAM reconstruction using ORB feature extraction, triangulation for a 3D point cloud, and terrain wireframe mapping.

- `gui.py`
  Hosts the PyQt5-based GUI for running pipeline steps, previewing results, and exporting outputs.

- `main.py`
  Acts as the command-line entry point to run individual pipeline stages (frames, depth, fusion, VP-SLAM, mesh, scene, physics, topology).

*Pro-tip:* The code is modular. Feel free to run each component individually or integrate them into your own custom pipeline.

---

## Usage

### Command-Line Interface

Run different pipeline steps using `main.py`. Here are some examples:

- **Extract Frames:**

  ```bash
  python main.py --step frames
  ```

- **Depth Estimation:**

  ```bash
  python main.py --step depth --frame_idx 5
  ```

- **Layered Segmentation:**

  ```bash
  python main.py --step layers --frame_idx 5
  ```

- **Multi-scale Segmentation:**

  ```bash
  python main.py --step multiscale --frame_idx 5
  ```

- **3D Terrain Mapping:**

  ```bash
  python main.py --step terrain --frame_idx 5
  ```

- **SLAM-Based Reconstruction:**

  ```bash
  python main.py --step slam
  ```

- **Physics Analysis & Video Creation:**

  The pipeline exposes a function `create_physics_video(start_frame, end_frame, output_path, base_dir, fps)`—check the code for details or run the corresponding stage via CLI.

*Remember:* Afaict, you can also batch process multiple frames by specifying `--start_frame` and `--end_frame`.

### GUI Interface

Launch the GUI if you prefer a visual control panel:

```bash
python gui.py
```

The GUI lets you:

- **Select pipeline step:** Frames, Depth, Layers, Multiscale, Terrain, or SLAM.
- **Specify frame range or a single frame.**
- **Configure parameters:** Batch size, number of layers.
- **Preview the output images or videos.**
- **Toggle between dark and light themes** (because we’re not boomer about style).
- **Export results** (placeholder functionality available).

Double-click the preview image to toggle zoom for that extra, “I need more detail” effect.

---

## Configuration

Paths and device settings are configured at the top of each module. Check the following variables:

- `FRAME_DIR`, `DEPTH_DIR`, `SEGMENTATION_DIR`, etc. — change these as needed.
- **Model Paths:** Ensure `MODEL_PATH`, `SAM_CHECKPOINT_PATH`, and `SAM_CONFIG_PATH` point to your downloaded model weights and config files.
- **DEVICE:** This will auto-select CUDA if available, but you can force CPU by changing the setting.

---

## Contributing

Contributions are welcome if you’re up to the challenge! Feel free to open issues, suggest improvements, or submit pull requests. Just remember: **no lazy shortcuts**—we expect code that’s as sharp as it is clever.

---

## License

This project is released under the [MIT License](LICENSE). Use, modify, and distribute at your own risk—no warranties provided, bc we’re all about experimentation and innovation.

---

## A Few Final Words

This pipeline is a **research-grade toolkit** intended for visionary developers and researchers who crave complete control and insight into 3D scene understanding and cinematic analysis.
Whether you’re automating pipelines, dissecting game physics, or just geeking out over scene graphs and SLAM integrations, this project is your playground.
So, buckle up, dive in, and enjoy automating like a boss—afaict, it’s gonna be lit.

---
