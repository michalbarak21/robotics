#!/usr/bin/env python
"""
Generate RGB‑D frames from a sub‑segment of a video with MiDaS and simulate a LiDAR
scan.  Additionally, create a side‑by‑side visualisation video showing:
    • Left  : original RGB frame
    • Middle: depth (Turbo colormap)
    • Right : LiDAR points from bird's eye view

Run:
    python simulate_rgbd_lidar.py \
        --video /path/to/input.mp4 \
        --start_sec 5 --end_sec 45 \
        --fps 5 \
        --num_beams 128 \
        --horizontal_res 100 \
        --max_depth 100.0 \
        --vis_out outputs/preview.mp4

Outputs (under ./outputs/ by default):
    rgb/frame_####.png          – original RGB
    depth/frame_####.npy        – float32 depth map
    lidar/scan_####.npy         – Nx3 float32 array of 3D points (x,y,z)
    preview.mp4                 – side‑by‑side visualisation
"""

from __future__ import annotations
import argparse, os, pathlib, cv2, torch, numpy as np, torchvision.transforms as T
from tqdm import tqdm

# ──────────────────────────────── CLI ─────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True, help="Input video path")
parser.add_argument("--fps", type=float, default=5.0, help="Target FPS to process & store")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--out_dir", default="outputs", help="Folder for RGB‑D & LiDAR data")
parser.add_argument("--vis_out", default="outputs/preview.mp4", help="Visualization video path")
parser.add_argument("--num_beams", type=int, default=128, help="Number of LiDAR beams to simulate")
parser.add_argument("--horizontal_res", type=int, default=100, help="Horizontal resolution of LiDAR")
parser.add_argument("--max_depth", type=float, default=100.0, help="Maximum depth for LiDAR points (meters)")
parser.add_argument("--disable_coloring", action="store_true", default=True, help="Disable coloring of LiDAR points")
parser.add_argument("--bev_range", type=float, default=50.0, help="Range of bird's eye view in meters")
args = parser.parse_args()

# ────────────────────────── Output folders ────────────────────────────────────
out_root = pathlib.Path(args.out_dir)
rgb_dir, depth_dir, lidar_dir = [out_root / p for p in ("rgb", "depth", "lidar")]
for p in (rgb_dir, depth_dir, lidar_dir): p.mkdir(parents=True, exist_ok=True)

# ───────────────────────── MiDaS DPT Large ────────────────────────────────────
print("Loading MiDaS (DPT_Large)…")
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type).to(args.device).eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# ───────────────────────────── Video IO ───────────────────────────────────────
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {args.video}")
fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0  # fallback
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
H_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

start_frame = 0  # int(args.start_sec * fps_in)
end_frame = frame_count - 1 # int(args.end_sec * fps_in) if args.end_sec else frame_count - 1
if start_frame >= frame_count:
    raise ValueError("start_sec beyond video length")

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
frame_interval = max(1, round(fps_in / args.fps))

# ──────────────── VideoWriter for visualisation ──────────────────────────────
vis_width = W_in * 3  # RGB | depth | lidar
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(args.vis_out), fourcc, args.fps, (vis_width, H_in))

# ─────────────── Function to convert depth to 3D points ─────────────────────
def depth_to_3d_points(depth_map, rows, cols, max_depth):
    """
    Convert depth map to 3D points.
    Returns array of (x,y,z) points and a mask of valid points.
    """
    # Create meshgrid for pixel coordinates
    v, u = rows, cols
    
    # Assuming simple pinhole camera model
    # fx, fy = focal lengths, cx, cy = principal points
    fx = fy = max(H_in, W_in)  # Approximation of focal length
    cx, cy = W_in / 2, H_in / 2  # Center of image
    
    # Get depth values at specified rows and columns
    z = depth_map[v, u]
    
    # Filter out points beyond max_depth or with invalid depth
    valid_mask = (z > 0) & (z <= max_depth)
    
    # Calculate 3D coordinates
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack to create Nx3 array
    points = np.stack([x, y, z], axis=-1)
    
    return points, valid_mask

# ──────── Function to create bird's eye view visualization ──────────────────
def create_birds_eye_view(points, img_height, img_width, bev_range=50.0):
    """
    Create a bird's eye view visualization of 3D points.
    
    Args:
        points: Nx3 array of 3D points (x,y,z)
        img_height, img_width: Dimensions of the output image
        bev_range: Range in meters to show in the bird's eye view
        
    Returns:
        Bird's eye view image
    """
    # Create empty image
    bev_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    if len(points) == 0:
        return bev_img
    
    # Extract x and z coordinates for top-down view
    x_coords = points[:, 0]  # Left-right in camera frame
    z_coords = points[:, 2]  # Depth into scene
    y_coords = points[:, 1]  # Height (up-down)
    
    # Define transformation from 3D space to image space
    # Center of image is (0,0) in world coordinates, bottom of image is max_range
    # Map x from [-bev_range/2, bev_range/2] to [0, img_width]
    # Map z from [0, bev_range] to [img_height, 0] (inverted, as +z goes into scene)
    
    # Scale and shift to fit within image bounds
    x_img = ((x_coords / bev_range) + 0.5) * img_width
    z_img = (1.0 - (z_coords / bev_range)) * img_height
    
    # Filter points that fall within image bounds
    in_bounds = (x_img >= 0) & (x_img < img_width) & (z_img >= 0) & (z_img < img_height)
    x_img = x_img[in_bounds].astype(int)
    z_img = z_img[in_bounds].astype(int)
    
    # Get y values for potential coloring (if not using plain white)
    y_filtered = y_coords[in_bounds]
    
    # Draw points onto BEV image
    for i in range(len(x_img)):
        if args.disable_coloring:
            color = (255, 255, 255)  # White in BGR
        else:
            # Color based on height (y-coordinate)
            # Normalize height to [0, 255]
            y_min, y_max = np.min(y_filtered), np.max(y_filtered)
            y_range = max(y_max - y_min, 1e-6)
            normalized_y = int(255 * (y_filtered[i] - y_min) / y_range)
            
            if normalized_y < 128:
                # Blue to cyan gradient
                b, g, r = 255, normalized_y * 2, 0
            else:
                # Cyan to white gradient
                b, g, r = 255, 255, (normalized_y - 128) * 2

            color = (b, g, r)
        
        # Draw point with a small radius for visibility
        cv2.circle(bev_img, (x_img[i], z_img[i]), 2, color, -1)
    
    # Add a reference grid
    grid_interval = img_height // 10
    for i in range(0, img_height, grid_interval):
        cv2.line(bev_img, (0, i), (img_width, i), (50, 50, 50), 1)
    for i in range(0, img_width, grid_interval):
        cv2.line(bev_img, (i, 0), (i, img_height), (50, 50, 50), 1)
    
    # Add a "car" indicator at the bottom center
    car_pos_x = img_width // 2
    car_pos_y = img_height - 20
    cv2.drawMarker(bev_img, (car_pos_x, car_pos_y), (0, 0, 255), cv2.MARKER_TRIANGLE_UP, 20, 2)
    
    # Add distance indicators
    meters_per_grid = bev_range / 10
    for i in range(1, 10):
        y_pos = img_height - int(i * grid_interval)
        text = f"{int(i * meters_per_grid)}m"
        cv2.putText(bev_img, text, (5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    return bev_img

# ─────────────────────────── Processing loop ─────────────────────────────────
frame_idx_global = start_frame
saved_idx = 0
pbar_total = (end_frame - start_frame) // frame_interval + 1
with tqdm(total=pbar_total, desc="Processing") as pbar:
    while frame_idx_global <= end_frame:
        ok, bgr = cap.read()
        if not ok:
            break
        if (frame_idx_global - start_frame) % frame_interval != 0:
            frame_idx_global += 1
            continue

        # RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Depth inference
        inp = transform(rgb).to(args.device)
        with torch.no_grad():
            depth_pred = midas(inp).squeeze().cpu().numpy()
        depth = cv2.resize(depth_pred, (W_in, H_in), interpolation=cv2.INTER_CUBIC)

        # LiDAR simulation with configurable beams and horizontal resolution
        vertical_indices = np.linspace(0, H_in - 1, args.num_beams).astype(int)
        horizontal_indices = np.linspace(0, W_in - 1, args.horizontal_res).astype(int)
        
        # Create meshgrid for all LiDAR points
        rows, cols = np.meshgrid(vertical_indices, horizontal_indices, indexing='ij')
        rows = rows.flatten()
        cols = cols.flatten()
        
        # Convert to 3D points
        points, valid_mask = depth_to_3d_points(depth, rows, cols, args.max_depth)
        
        # Filter valid points only
        valid_points = points[valid_mask]
        valid_rows = rows[valid_mask]
        valid_cols = cols[valid_mask]

        # ─────────── Persist numpy & images ───────────
        cv2.imwrite(str(rgb_dir / f"frame_{saved_idx:05d}.png"), bgr)
        np.save(depth_dir / f"frame_{saved_idx:05d}.npy", depth.astype(np.float32))
        np.save(lidar_dir / f"scan_{saved_idx:05d}.npy", valid_points.astype(np.float32))

        # ─────────── Build visualisation frame ─────────
        # Depth colourmap
        d_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_vis = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)

        # LiDAR visualization as bird's eye view
        lidar_vis = create_birds_eye_view(valid_points, H_in, W_in, args.bev_range)
        
        # Stack & write to video
        side_by_side = cv2.hconcat([bgr, depth_vis, lidar_vis])
        writer.write(side_by_side)
        
        # Print the number of valid points for this frame
        print(f"Frame {saved_idx}: {len(valid_points)} valid LiDAR points")

        saved_idx += 1
        frame_idx_global += 1
        pbar.update(1)

cap.release()
writer.release()
print(f"\n✔ Saved {saved_idx} processed frames")
print(f"  • RGB       -> {rgb_dir}")
print(f"  • Depth     -> {depth_dir}")
print(f"  • LiDAR     -> {lidar_dir} ({len(valid_points) if 'valid_points' in locals() else 0} points in last frame)")
print(f"  • Preview   -> {pathlib.Path(args.vis_out)}")
  