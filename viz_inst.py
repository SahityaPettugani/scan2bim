import numpy as np
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
import pyransac3d as pyrsc
from sklearn.neighbors import NearestNeighbors # Added for smoothing

import torch
torch.backends.cudnn.benchmark = True
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from model.segcloud import SegCloud
from model.bimnet import BIMNet
from dataloaders.PCSdataset import PCSDataset
from dataloaders.S3DISdataset import S3DISDataset

from sklearn.cluster import DBSCAN
import json
from pathlib import Path
from matplotlib.colors import to_rgb
import argparse
import re
from datetime import datetime
import math

ID_TO_NAME = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
}

def load_point_cloud(file_path):
    print(f"Loading point cloud from: {file_path}")
    
    if file_path.suffix in ['.ply', '.pcd']:
        pcd = o3d.io.read_point_cloud(str(file_path))
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    print(f"Loaded {len(pcd.points)} points")
    return pcd

ID_TO_NAME = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    # 7: "unassigned",
}

def separate_by_label(pcd, point_labels):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    separated = {}
    for class_id, class_name in ID_TO_NAME.items():
        mask = (point_labels == class_id)
        if not np.any(mask):
            continue

        class_pcd = o3d.geometry.PointCloud()
        class_pcd.points = o3d.utility.Vector3dVector(points[mask])
        class_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

        separated[class_name] = class_pcd
        print(f"  {class_name}: {mask.sum()} points")

    return separated

def smooth_labels_knn(pcd, labels, k=5):
    print(f"Smoothing labels with KNN (k={k})...")
    points = np.asarray(pcd.points)

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', n_jobs=-1).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    new_labels = np.zeros_like(labels)
    
    neighbor_labels = labels[indices] 
    
    from scipy.stats import mode
    try:
        vote_result = mode(neighbor_labels, axis=1, keepdims=False)
        new_labels = vote_result[0]
    except:
        for i in tqdm(range(len(labels)), desc="Voting"):
            counts = np.bincount(neighbor_labels[i])
            new_labels[i] = np.argmax(counts)
            
    return new_labels

def apply_geometric_priors_to_labels(
    pcd,
    labels,
    wall_max_abs_nz=0.35,
    horiz_min_abs_nz=0.75,
    floor_quantile=0.35,
    ceiling_quantile=0.75,
    normal_k=30,
):
    """
    Geometry-guided label correction.
    - Reassign wall points with horizontal-like normals to floor/ceiling by z.
    - Reassign floor/ceiling points with vertical-like normals to wall.
    """
    if len(pcd.points) == 0:
        return labels

    pcd_normals = o3d.geometry.PointCloud(pcd)
    pcd_normals.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=normal_k)
    )
    normals = np.asarray(pcd_normals.normals)
    points = np.asarray(pcd.points)
    z = points[:, 2]

    z_floor_th = float(np.quantile(z, floor_quantile))
    z_ceil_th = float(np.quantile(z, ceiling_quantile))

    out = labels.copy()
    abs_nz = np.abs(normals[:, 2])

    # IDs from ID_TO_NAME map:
    # 0: ceiling, 1: floor, 2: wall
    wall_mask = (out == 2)
    floor_mask = (out == 1)
    ceil_mask = (out == 0)

    # Wall points that are too horizontal are likely floor/ceiling bleed.
    wall_to_horiz = wall_mask & (abs_nz >= horiz_min_abs_nz)
    out[wall_to_horiz & (z <= z_floor_th)] = 1
    out[wall_to_horiz & (z >= z_ceil_th)] = 0
    out[wall_to_horiz & (z > z_floor_th) & (z < z_ceil_th)] = 1

    # Floor/ceiling points with too vertical normals are likely wall bleed.
    floor_to_wall = floor_mask & (abs_nz <= wall_max_abs_nz)
    ceil_to_wall = ceil_mask & (abs_nz <= wall_max_abs_nz)
    out[floor_to_wall | ceil_to_wall] = 2

    changed = int(np.sum(out != labels))
    print(f"Applied geometric priors on labels. Reassigned points: {changed}")
    return out

def instantiate_with_dbscan(pcd, class_name, eps=0.1, min_points=100):
    if len(pcd.points) == 0:
        return []
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    print(f"\nClustering {class_name} with DBSCAN...")
    # import matplotlib.pyplot as plt

    # # Calculate distances to the k-th nearest neighbor  
    # from sklearn.neighbors import NearestNeighbors
    # neigh = NearestNeighbors(n_neighbors=min_points)
    # nbrs = neigh.fit(points)
    # distances, indices = nbrs.kneighbors(points)
    # distances = np.sort(distances[:, -1], axis=0)

    # plt.plot(distances)
    # plt.ylabel("Epsilon distance")
    # plt.show()
    clustering = DBSCAN(eps=eps, min_samples=min_points, n_jobs=-1).fit(points)
    labels = clustering.labels_
    
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    
    print(f"  Found {n_clusters} instances")
    
    instances = []
    for label_id in unique_labels:
        if label_id == -1:
            continue
        
        instance_mask = labels == label_id
        instance_points = points[instance_mask]
        instance_colors = colors[instance_mask]
        
        instance_pcd = o3d.geometry.PointCloud()
        instance_pcd.points = o3d.utility.Vector3dVector(instance_points)
        instance_pcd.colors = o3d.utility.Vector3dVector(instance_colors)
        
        instances.append(instance_pcd)
    
    return instances

def filter_small_instances(instances_dict, min_points_thresholds):
    cleaned_dict = {}
    print("\n--- CLEANING NOISE ---")
    
    for class_name, instances in instances_dict.items():
        thresh = min_points_thresholds.get(class_name, 500)
        
        valid_instances = []
        for i, pcd in enumerate(instances):
            n_points = len(pcd.points)
            if n_points >= thresh:
                valid_instances.append(pcd)
        
        cleaned_dict[class_name] = valid_instances
        removed = len(instances) - len(valid_instances)
        if removed > 0:
            print(f"  {class_name}: Removed {removed} small instances (<{thresh} pts)")
            
    return cleaned_dict

def save_instances(instances_dict, output_dir):
    """
    Save all instantiated point clouds to separate files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for class_name, instances in instances_dict.items():
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        for i, instance in enumerate(instances):
            filename = class_dir / f"{class_name}_instance_{i:03d}.ply"
            o3d.io.write_point_cloud(str(filename), instance)
        
    summary = {
        class_name: len(instances) 
        for class_name, instances in instances_dict.items()
    }
    
    with open(output_path / "instantiation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {output_path / 'instantiation_summary.json'}")

    combined_pc = o3d.geometry.PointCloud()
    for class_name, instances in instances_dict.items():
        for instance in instances:
            combined_pc += instance   # merge point clouds
    
    combined_filename = output_path / "all_instances_combined.ply"
    o3d.io.write_point_cloud(str(combined_filename), combined_pc)
    print(f"Combined point cloud saved to {combined_filename}")

def generate_distinct_colors(n_colors):
    try: 
        cmap = plt.colormaps['tab20']
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap('tab20')
    
    colors = []
    for i in range(n_colors):
        rgba = cmap(i / max(n_colors, 1))
        colors.append(rgba[:3]) 
    return colors

def visualize_instances(instances_dict, show_by_class=True):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    
    def show_instance_legend(instance_labels, instance_colors):
        import math
        patches = [mpatches.Patch(color=instance_colors[i], label=instance_labels[i]) for i in range(len(instance_labels))]
        plt.figure(figsize=(min(12, max(6, len(instance_labels)//2)), 2 + math.ceil(len(instance_labels)/6)))
        plt.legend(handles=patches, loc='center', ncol=6, fontsize=10, frameon=False)
        plt.axis('off')
        plt.title('Instance-Color Legend')
        plt.show(block=False)
        input("Legend displayed. Press Enter to continue...")

    # Always show instance-level legend
    instance_labels = []
    instance_colors = []
    all_colored_instances = []
    for class_name, instances in instances_dict.items():
        if len(instances) == 0:
            continue
        colors = generate_distinct_colors(len(instances))
        for i, instance in enumerate(instances):
            label = f"{class_name} {i}"
            instance_labels.append(label)
            instance_colors.append(colors[i])
            colored_pcd = o3d.geometry.PointCloud(instance)
            instance_color = np.tile(colors[i], (len(instance.points), 1))
            colored_pcd.colors = o3d.utility.Vector3dVector(instance_color)
            all_colored_instances.append(colored_pcd)
    if instance_labels:
        show_instance_legend(instance_labels, instance_colors)
    if all_colored_instances:
        o3d.visualization.draw_geometries(all_colored_instances,
                                        window_name=f"All Instances",
                                        width=1024, height=768)

def visualize_summary(instances_dict, separated_classes, original_pcd):
    print("\n" + "=" * 60)
    print("VISUALIZATION MODE")
    print("=" * 60)
    
    if separated_classes:
        o3d.visualization.draw_geometries(list(separated_classes.values()), window_name="Semantic Classes", width=800, height=600)
    
    if instances_dict:
        visualize_instances(instances_dict, show_by_class=False)

    print("\n" + "=" * 60)
    response = input("Would you like to see all instances from all classes separately? (y/n): ")
    if response.lower() == 'y':
        visualize_instances(instances_dict, show_by_class=True)

def finetune_model(checkpoint_path, device, num_old_classes, num_new_classes):
    state_old = torch.load(checkpoint_path, map_location=device)
    model_new = BIMNet(num_classes=num_new_classes)
    state_new = model_new.state_dict()

    transferred, skipped = [], []
    for k, v in state_old.items():
        if k in state_new and state_new[k].shape == v.shape:
            state_new[k] = v
            transferred.append(k)
        else:
            skipped.append(k)
    
    if skipped:
        print("Skipped parameters:")
        for k in skipped:
            print(f" - {k} : {state_old[k].shape}")

    model_new.load_state_dict(state_new)
    model_new.to(device)
    model_new.train()

    return model_new

def build_models(checkpoint_paths, device, num_classes=7):
    models = []
    for ckpt in checkpoint_paths:
        print(f"Loading checkpoint: {ckpt}")
        model = finetune_model(ckpt, device, num_old_classes=13, num_new_classes=num_classes)
        model.eval()
        models.append(model)
    return models

# def voxelize_points(points, cube_edge):
#     points_centered = points - points.mean(axis=0)
    
#     max_val = np.abs(points_centered).max() + 1e-8
#     points_norm = points_centered / max_val
    
#     points_shifted = points_norm + 1.0
    
#     scale_factor = cube_edge // 2
#     points_grid = np.round(points_shifted * scale_factor).astype(np.int32)
    
#     points_grid = np.clip(points_grid, 0, cube_edge - 1)

#     vox = np.zeros((1, cube_edge, cube_edge, cube_edge), dtype=np.float32)
#     vox[0, points_grid[:, 0], points_grid[:, 1], points_grid[:, 2]] = 1.0

#     return vox, points_grid

def voxelize_points(points, cube_edge):
    points_centered = points - points.mean(axis=0)
    
    points_centered[:, 2] -= points_centered[:, 2].min()
    
    ranges = points_centered.max(axis=0) - points_centered.min(axis=0)
    max_dim = ranges.max() + 1e-6
    scale_factor = 1.8 / max_dim
    
    points_norm = points_centered * scale_factor
    
    points_shifted = points_norm 
    points_shifted[:, 2] -= 0.9 

    points_shifted += 1.0 
    points_grid = np.round(points_shifted * (cube_edge // 2)).astype(np.int32)
    
    points_grid = np.clip(points_grid, 0, cube_edge - 1)

    vox = np.zeros((1, cube_edge, cube_edge, cube_edge), dtype=np.float32)
    vox[0, points_grid[:, 0], points_grid[:, 1], points_grid[:, 2]] = 1.0

    return vox, points_grid

def color_label(labels, num_classes=7):
    cmap = plt.get_cmap("tab20", num_classes)
    flat = labels.flatten()
    colors = cmap(flat % num_classes)[:, :3] 
    return colors.reshape((*labels.shape, 3))

def run_bimnet_inference(pcd, models, cube_edge=96, num_classes=7, device="cuda"):
    points = np.asarray(pcd.points)
    print(f"Loaded {points.shape[0]} points")
    vox, points_grid = voxelize_points(points, cube_edge)
    x = torch.from_numpy(vox).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_sum = None
        for model in models:
            logits = model(x)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        logits_avg = logits_sum / len(models)
        preds = logits_avg.argmax(dim=1).squeeze(0).cpu().numpy()

    colors_volume = color_label(preds, num_classes=num_classes)
    point_colors = colors_volume[points_grid[:, 0], points_grid[:, 1], points_grid[:, 2]]
    point_labels = preds[points_grid[:, 0], points_grid[:, 1], points_grid[:, 2]]

    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    return pcd, preds, points_grid, point_labels

def instantiate_planar_iterative(
    pcd,
    class_name,
    dist_thresh=0.20,
    min_points=500,
    ransac_min_points=100,
    ransac_max_iterations=1000,
    wall_vertical_tol=0.25,
    horizontal_min_alignment=0.85,
    wall_min_height=1.8,
    wall_min_length=1.0,
    wall_max_width=1.2,
):
    """
    Separates planar instances (Walls/Floors) by iteratively finding planes 
    and removing them from the cloud until no valid planes remain.
    UPDATED: dist_thresh increased to 0.20 to fix fragmented walls.
    """
    remaining_pcd = pcd
    instances = []
    
    print(f"\nIterative RANSAC for {class_name} (Thresh={dist_thresh})...")
    
    while len(remaining_pcd.points) > min_points:
        points = np.asarray(remaining_pcd.points)

        plane = pyrsc.Plane()
        # Note: pyransac3d returns equation (4 floats) and inliers (indices)
        best_eq, inliers = plane.fit(points, thresh=dist_thresh, minPoints=100, maxIteration=1000)
        
        if len(inliers) < min_points:
            break
            
        # Extract the instance
        inst_pcd = remaining_pcd.select_by_index(inliers)

        # Additional wall-shape filtering to suppress false wall planes.
        if class_name == "wall":
            inst_pts = np.asarray(inst_pcd.points)
            z_extent = float(inst_pts[:, 2].max() - inst_pts[:, 2].min())

            xy = inst_pts[:, :2]
            xy_centered = xy - xy.mean(axis=0)
            _, _, vh = np.linalg.svd(xy_centered, full_matrices=False)
            axis1 = vh[0]
            axis2 = vh[1]
            proj1 = xy_centered @ axis1
            proj2 = xy_centered @ axis2
            length_extent = float(proj1.max() - proj1.min())
            width_extent = float(proj2.max() - proj2.min())

            shape_ok = (
                z_extent >= wall_min_height
                and length_extent >= wall_min_length
                and width_extent <= wall_max_width
            )
            if not shape_ok:
                print(
                    f"  Rejected wall plane by shape: "
                    f"height={z_extent:.2f}, length={length_extent:.2f}, width={width_extent:.2f} "
                    f"(need h>={wall_min_height}, l>={wall_min_length}, w<={wall_max_width})"
                )
                remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
                continue

        inst_pcd.paint_uniform_color(generate_distinct_colors(len(instances)+1)[-1])
        instances.append(inst_pcd)

        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
        print(f"  Found instance {len(instances)}: {len(inliers)} points. Remaining: {len(remaining_pcd.points)}")
        
    return instances

def extract_bim_parameters(instances_dict):
    """
    Calculates BIM-ready parameters (Length, Height, Thickness, Centerline) 
    for each wall instance.
    """
    def q(arr, lo=0.02, hi=0.98):
        return float(np.quantile(arr, lo)), float(np.quantile(arr, hi))

    bim_data = []

    # Build robust global envelope from walls to anchor slabs.
    wall_pts_all = []
    for wall_pcd in instances_dict.get("wall", []):
        pts = np.asarray(wall_pcd.points)
        if len(pts) >= 500:
            wall_pts_all.append(pts)

    if wall_pts_all:
        wall_all = np.vstack(wall_pts_all)
        wx0, wx1 = q(wall_all[:, 0], 0.02, 0.98)
        wy0, wy1 = q(wall_all[:, 1], 0.02, 0.98)
        wz0, wz1 = q(wall_all[:, 2], 0.02, 0.98)
    else:
        wx0 = wx1 = wy0 = wy1 = wz0 = wz1 = None

    for class_name, pcd_list in instances_dict.items():
        for idx, pcd in enumerate(pcd_list):
            pts = np.asarray(pcd.points)
            if len(pts) < 50:
                continue

            # 1. Height (Z-axis extent)
            z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
            height = z_max - z_min
            
            if class_name in ["floor", "ceiling"]:
                    # Floors/Ceilings are horizontal slabs. Centerlines don't make sense.
                    # Use a bounding box approach instead.
                    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
                    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
                    
                    bim_obj = {
                        "id": f"{class_name}_{idx}",
                        "type": class_name,
                        "height": float(height), # Will be close to 0
                        "thickness": 0.2, # Standard slab thickness
                        "geometry": {
                            "start_x": float(x_min), "start_y": float(y_min), "start_z": float(z_min),
                            "end_x": float(x_max), "end_y": float(y_max), "end_z": float(z_min)
                        }
                    }
            else:
                xy_pts = pts[:, :2]
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                pca.fit(xy_pts)

                direction = pca.components_[0]
                direction = direction / (np.linalg.norm(direction) + 1e-12)
                normal2d = np.array([-direction[1], direction[0]])
                center = xy_pts.mean(axis=0)

                proj_main = xy_pts @ direction
                p_min, p_max = np.quantile(proj_main, 0.02), np.quantile(proj_main, 0.98)
                start_pt = center + direction * (p_min - proj_main.mean())
                end_pt = center + direction * (p_max - proj_main.mean())

                proj_side = (xy_pts - center) @ normal2d
                thick_q = np.quantile(proj_side, 0.95) - np.quantile(proj_side, 0.05)
                thickness = float(max(0.08, min(0.8, thick_q)))

                bim_obj = {
                    "id": f"{class_name}_{idx}",
                    "type": class_name,
                    "height": float(height),
                    "thickness": 0.2, # Consider dynamically calculating this later
                    "geometry": {
                        "start_x": float(start_pt[0]), "start_y": float(start_pt[1]), "start_z": float(z0),
                        "end_x": float(end_pt[0]), "end_y": float(end_pt[1]), "end_z": float(z0)
                    }
                }

            bim_data.append(bim_obj)

    return bim_data


def _to_xy(v):
    return np.array([float(v[0]), float(v[1])], dtype=float)


def _segment_dir(a, b):
    d = b - a
    n = np.linalg.norm(d) + 1e-12
    return d / n


def _segment_length(a, b):
    return float(np.linalg.norm(b - a))


def _line_intersection(p1, p2, q1, q2):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return np.array([px, py], dtype=float)


def _point_to_segment_distance(p, a, b):
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _point_to_line_distance(p, a, d_unit):
    # distance to infinite line through a in direction d_unit
    v = p - a
    proj = np.dot(v, d_unit) * d_unit
    return float(np.linalg.norm(v - proj))


def _projection_interval(a, b, origin, d_unit):
    ta = float(np.dot(a - origin, d_unit))
    tb = float(np.dot(b - origin, d_unit))
    return min(ta, tb), max(ta, tb)


def _angle_parallel(d1, d2, tol_deg=10.0):
    c = abs(float(np.dot(d1, d2)))
    c = min(1.0, max(-1.0, c))
    ang = math.degrees(math.acos(c))
    return ang <= tol_deg


def _merge_collinear_walls(walls, angle_tol_deg=8.0, offset_tol=0.2, gap_tol=0.4):
    merged = []
    used = [False] * len(walls)

    for i, wi in enumerate(walls):
        if used[i]:
            continue
        group_idx = [i]
        used[i] = True

        ai = _to_xy([wi["geometry"]["start_x"], wi["geometry"]["start_y"]])
        bi = _to_xy([wi["geometry"]["end_x"], wi["geometry"]["end_y"]])
        di = _segment_dir(ai, bi)

        for j in range(i + 1, len(walls)):
            if used[j]:
                continue
            wj = walls[j]
            aj = _to_xy([wj["geometry"]["start_x"], wj["geometry"]["start_y"]])
            bj = _to_xy([wj["geometry"]["end_x"], wj["geometry"]["end_y"]])
            dj = _segment_dir(aj, bj)
            if not _angle_parallel(di, dj, tol_deg=angle_tol_deg):
                continue
            # close enough to same line
            if _point_to_line_distance(aj, ai, di) > offset_tol and _point_to_line_distance(bj, ai, di) > offset_tol:
                continue

            i0, i1 = _projection_interval(ai, bi, ai, di)
            j0, j1 = _projection_interval(aj, bj, ai, di)
            separated_gap = max(i0, j0) - min(i1, j1)
            if separated_gap > gap_tol:
                continue
            used[j] = True
            group_idx.append(j)

        # merge group on dominant line
        if len(group_idx) == 1:
            merged.append(wi)
            continue

        pts = []
        heights = []
        thicknesses = []
        z_vals = []
        for k in group_idx:
            w = walls[k]
            a = _to_xy([w["geometry"]["start_x"], w["geometry"]["start_y"]])
            b = _to_xy([w["geometry"]["end_x"], w["geometry"]["end_y"]])
            pts.extend([a, b])
            heights.append(float(w.get("height", 0.0)))
            thicknesses.append(float(w.get("thickness", 0.2)))
            z_vals.append(float(w["geometry"].get("start_z", 0.0)))

        ts = [float(np.dot(p - ai, di)) for p in pts]
        tmin, tmax = min(ts), max(ts)
        s = ai + tmin * di
        e = ai + tmax * di

        w_base = dict(wi)
        w_base["height"] = max(heights) if heights else w_base.get("height", 0.0)
        w_base["thickness"] = float(np.median(thicknesses)) if thicknesses else w_base.get("thickness", 0.2)
        z_base = float(np.median(z_vals)) if z_vals else float(w_base["geometry"].get("start_z", 0.0))
        w_base["geometry"] = {
            "start_x": float(s[0]), "start_y": float(s[1]), "start_z": z_base,
            "end_x": float(e[0]), "end_y": float(e[1]), "end_z": z_base
        }
        merged.append(w_base)

    return merged


def _snap_wall_endpoints_to_intersections(walls, endpoint_snap_tol=0.45, line_proximity_tol=0.25):
    segs = []
    for w in walls:
        a = _to_xy([w["geometry"]["start_x"], w["geometry"]["start_y"]])
        b = _to_xy([w["geometry"]["end_x"], w["geometry"]["end_y"]])
        segs.append((a, b))

    for i in range(len(walls)):
        ai, bi = segs[i]
        best_start = None
        best_end = None
        best_ds = 1e18
        best_de = 1e18
        for j in range(len(walls)):
            if i == j:
                continue
            aj, bj = segs[j]
            inter = _line_intersection(ai, bi, aj, bj)
            if inter is None:
                continue
            # intersection should be near both finite segments
            if _point_to_segment_distance(inter, ai, bi) > line_proximity_tol:
                continue
            if _point_to_segment_distance(inter, aj, bj) > line_proximity_tol:
                continue

            ds = float(np.linalg.norm(inter - ai))
            de = float(np.linalg.norm(inter - bi))
            if ds < best_ds and ds <= endpoint_snap_tol:
                best_ds = ds
                best_start = inter
            if de < best_de and de <= endpoint_snap_tol:
                best_de = de
                best_end = inter

        if best_start is not None:
            walls[i]["geometry"]["start_x"] = float(best_start[0])
            walls[i]["geometry"]["start_y"] = float(best_start[1])
        if best_end is not None:
            walls[i]["geometry"]["end_x"] = float(best_end[0])
            walls[i]["geometry"]["end_y"] = float(best_end[1])

    return walls


def refine_wall_geometry(
    bim_data,
    merge_angle_tol_deg=8.0,
    merge_offset_tol=0.2,
    merge_gap_tol=0.4,
    endpoint_snap_tol=0.45,
    line_proximity_tol=0.25,
):
    walls = [x for x in bim_data if x.get("type") == "wall"]
    others = [x for x in bim_data if x.get("type") != "wall"]
    if not walls:
        return bim_data

    before = len(walls)
    walls = _merge_collinear_walls(
        walls,
        angle_tol_deg=merge_angle_tol_deg,
        offset_tol=merge_offset_tol,
        gap_tol=merge_gap_tol,
    )
    walls = _snap_wall_endpoints_to_intersections(
        walls,
        endpoint_snap_tol=endpoint_snap_tol,
        line_proximity_tol=line_proximity_tol,
    )
    after = len(walls)
    print(f"Wall refinement: {before} -> {after} wall segments after merge/snap.")
    return others + walls

def main(
    input_file,
    output_dir="output_instances",
    checkpoint_paths=None,
    cube_edge=96,
    num_classes=7,
    device=None,
    visualize_network_output=False,
    visualize_instances_flag=True,
    enable_smoothing=True,
    smooth_k=5,
    smooth_max_points=500000,
    strong_smooth_k=15,
    planar_dist_thresh=0.15,
    planar_min_points=2000,
    ransac_min_points=100,
    ransac_max_iterations=1000,
    wall_vertical_tol=0.25,
    horizontal_min_alignment=0.85,
    wall_min_height=1.8,
    wall_min_length=1.0,
    wall_max_width=1.2,
    denoise=True,
    denoise_nb_neighbors=30,
    denoise_std_ratio=2.0,
    refine_walls=True,
    merge_angle_tol_deg=8.0,
    merge_offset_tol=0.2,
    merge_gap_tol=0.4,
    endpoint_snap_tol=0.45,
    line_proximity_tol=0.25,
    apply_geometric_priors=True,
    wall_max_abs_nz=0.35,
    horiz_min_abs_nz=0.75,
    floor_quantile=0.35,
    ceiling_quantile=0.75,
    normal_k=30,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_paths = checkpoint_paths 
    
    print("=" * 60)
    print("Point Cloud Instantiation Workflow (BIMNet + DBSCAN)")
    print("=" * 60)

    input_path = Path(input_file)
    pcd = load_point_cloud(input_path)
    if denoise:
        pcd = denoise_point_cloud(
            pcd,
            nb_neighbors=denoise_nb_neighbors,
            std_ratio=denoise_std_ratio,
        )
    run_output_dir = create_run_output_dir(output_dir, input_path)
    print(f"Run output directory: {run_output_dir}")

    print("\nLoading BIMNet models...")
    models = build_models(checkpoint_paths, device, num_classes=num_classes)

    pcd, preds_volume, points_grid, point_labels = run_bimnet_inference(
        pcd, models, cube_edge=cube_edge, num_classes=num_classes, device=device
    )

    # --- NEW STEP: SMOOTH LABELS ---
    # Fixes salt-and-pepper noise before any separation happens
    print("\nStep 0.5: Smoothing predictions with KNN...")
    point_labels = smooth_labels_knn(pcd, point_labels, k=5)
    
    print("\nStep 1: Separating point cloud by semantic class...")
    separated_classes = separate_by_label(pcd, point_labels)

    if not separated_classes:
        print("Warning: No classes found! Check your color mappings.")
        return None

    print("\nStep 2: Instantiating classes...")
    all_instances = {}

    planar_classes = ['wall', 'floor', 'ceiling']
    dbscan_params = {
        'beam':      {'eps': 0.3, 'min_points': 100},
        'column':    {'eps': 0.3, 'min_points': 100},
        'window':    {'eps': 0.2, 'min_points': 50},
        'door':      {'eps': 0.3, 'min_points': 100},
    }

    # Apply stronger smoothing before instance extraction
    if enable_smoothing:
        print(f"Applying strong smoothing (KNN, k={strong_smooth_k})...")
        point_labels = smooth_labels_knn(pcd, point_labels, k=strong_smooth_k)
        separated_classes = separate_by_label(pcd, point_labels)

    for class_name, class_pcd in separated_classes.items():
        if class_name in planar_classes:
            # UPDATED: Thresh 0.20 handles wavy walls
            instances = instantiate_planar_iterative(class_pcd, class_name, dist_thresh=0.05)
        else:
            params = dbscan_params.get(class_name, {'eps': 0.1, 'min_points': 100})
            instances = instantiate_with_dbscan(
                class_pcd,
                class_name,
                eps=params['eps'],
                min_points=params['min_points'],
            )
        all_instances[class_name] = instances

    cleaning_thresholds = {
        'ceiling': 2000, 
        'floor': 2000,   
        'wall': 1000,
        'beam': 50,
        'column': 50,
        'window': 20,
        'door': 50, 
    }

    all_instances = filter_small_instances(all_instances, cleaning_thresholds)

    print("\nStep 3: Extracting BIM Parameters and Saving...")
    save_instances(all_instances, run_output_dir)
    
    bim_json_data = extract_bim_parameters(all_instances)
    if refine_walls:
        bim_json_data = refine_wall_geometry(
            bim_json_data,
            merge_angle_tol_deg=merge_angle_tol_deg,
            merge_offset_tol=merge_offset_tol,
            merge_gap_tol=merge_gap_tol,
            endpoint_snap_tol=endpoint_snap_tol,
            line_proximity_tol=line_proximity_tol,
        )
    with open(Path(run_output_dir) / "bim_reconstruction_data.json", "w") as f:
        json.dump(bim_json_data, f, indent=4)
    print(f"BIM parameters saved to {run_output_dir}/bim_reconstruction_data.json")

    if visualize_instances_flag:
        print("\nStep 4: Visualizing extracted instances...")
        visualize_instances(all_instances, show_by_class=False)

    return all_instances, separated_classes, pcd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BIMNet semantic segmentation + DBSCAN instance extraction"
    )
    parser.add_argument("--input_file", required=True, help="Path to input point cloud (.ply/.pcd)")
    parser.add_argument("--output_dir", default="output_instances", help="Root output directory (a new subfolder is created per run)")
    parser.add_argument("--checkpoint", action="append", required=True, help="Path(s) to BIMNet checkpoint(s)")
    parser.add_argument("--cube_edge", type=int, default=96, help="Voxel grid edge length")
    parser.add_argument("--num_classes", type=int, default=7, help="Number of BIMNet output classes")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--vis-net", action="store_true", help="Visualize BIMNet output")
    parser.add_argument("--vis-instances", action="store_true", help="Visualize DBSCAN instances (legacy flag; visualization is already on by default)")
    parser.add_argument("--no-vis-instances", action="store_true", help="Disable automatic instance visualization")
    parser.add_argument("--no-denoise", action="store_true", help="Disable statistical outlier removal before inference")
    parser.add_argument("--denoise-nb-neighbors", type=int, default=30, help="Neighbors for outlier removal")
    parser.add_argument("--denoise-std-ratio", type=float, default=2.0, help="Std ratio for outlier removal (smaller is stronger)")
    parser.add_argument("--no-refine-walls", action="store_true", help="Disable post-processing to merge/snap wall geometry")
    parser.add_argument("--merge-angle-tol-deg", type=float, default=8.0, help="Max angle difference for merging near-collinear walls")
    parser.add_argument("--merge-offset-tol", type=float, default=0.2, help="Max lateral offset for merging wall centerlines")
    parser.add_argument("--merge-gap-tol", type=float, default=0.4, help="Max projected gap for merging adjacent wall segments")
    parser.add_argument("--endpoint-snap-tol", type=float, default=0.45, help="Max endpoint move when snapping wall joints")
    parser.add_argument("--line-proximity-tol", type=float, default=0.25, help="Intersection must be this close to both segments to snap")
    parser.add_argument("--no-geo-priors", action="store_true", help="Disable geometric prior relabeling before instantiation")
    parser.add_argument("--wall-max-abs-nz", type=float, default=0.35, help="Max |normal_z| for wall-like surface")
    parser.add_argument("--horiz-min-abs-nz", type=float, default=0.75, help="Min |normal_z| for horizontal-like surface")
    parser.add_argument("--floor-quantile", type=float, default=0.35, help="Z quantile threshold used as floor band")
    parser.add_argument("--ceiling-quantile", type=float, default=0.75, help="Z quantile threshold used as ceiling band")
    parser.add_argument("--normal-k", type=int, default=30, help="KNN size for normal estimation in geometric priors")
    parser.add_argument("--no-smooth", action="store_true", help="Disable KNN label smoothing")
    parser.add_argument("--smooth-k", type=int, default=5, help="K for KNN smoothing")
    parser.add_argument(
        "--strong-smooth-k",
        type=int,
        default=15,
        help="K for second, stronger KNN smoothing pass before instantiation",
    )
    parser.add_argument(
        "--smooth-max-points",
        type=int,
        default=500000,
        help="Auto-skip KNN smoothing if point count exceeds this threshold",
    )
    parser.add_argument(
        "--planar-dist-thresh",
        type=float,
        default=0.15,
        help="RANSAC inlier distance threshold for wall/floor/ceiling instantiation",
    )
    parser.add_argument(
        "--planar-min-points",
        type=int,
        default=2000,
        help="Minimum inlier count to accept one planar instance",
    )
    parser.add_argument(
        "--ransac-min-points",
        type=int,
        default=100,
        help="Minimum points used by pyransac3d for each plane fit",
    )
    parser.add_argument(
        "--ransac-max-iterations",
        type=int,
        default=1000,
        help="Maximum iterations for each plane fit",
    )
    parser.add_argument(
        "--wall-vertical-tol",
        type=float,
        default=0.25,
        help="For wall planes, max allowed |normal_z| (smaller is stricter)",
    )
    parser.add_argument(
        "--horizontal-min-alignment",
        type=float,
        default=0.85,
        help="For floor/ceiling planes, min required |normal_z| (larger is stricter)",
    )
    parser.add_argument(
        "--wall-min-height",
        type=float,
        default=1.8,
        help="Minimum vertical extent for a wall instance",
    )
    parser.add_argument(
        "--wall-min-length",
        type=float,
        default=1.0,
        help="Minimum XY length extent for a wall instance",
    )
    parser.add_argument(
        "--wall-max-width",
        type=float,
        default=1.2,
        help="Maximum XY width extent for a wall instance",
    )
    
    args = parser.parse_args()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    main(
        input_file=args.input_file,
        output_dir=args.output_dir,
        checkpoint_paths=args.checkpoint,
        cube_edge=args.cube_edge,
        num_classes=args.num_classes,
        device=device,
        visualize_network_output=args.vis_net,
        visualize_instances_flag=(not args.no_vis_instances) or args.vis_instances,
        enable_smoothing=not args.no_smooth,
        smooth_k=args.smooth_k,
        smooth_max_points=args.smooth_max_points,
        strong_smooth_k=args.strong_smooth_k,
        planar_dist_thresh=args.planar_dist_thresh,
        planar_min_points=args.planar_min_points,
        ransac_min_points=args.ransac_min_points,
        ransac_max_iterations=args.ransac_max_iterations,
        wall_vertical_tol=args.wall_vertical_tol,
        horizontal_min_alignment=args.horizontal_min_alignment,
        wall_min_height=args.wall_min_height,
        wall_min_length=args.wall_min_length,
        wall_max_width=args.wall_max_width,
        denoise=not args.no_denoise,
        denoise_nb_neighbors=args.denoise_nb_neighbors,
        denoise_std_ratio=args.denoise_std_ratio,
        refine_walls=not args.no_refine_walls,
        merge_angle_tol_deg=args.merge_angle_tol_deg,
        merge_offset_tol=args.merge_offset_tol,
        merge_gap_tol=args.merge_gap_tol,
        endpoint_snap_tol=args.endpoint_snap_tol,
        line_proximity_tol=args.line_proximity_tol,
        apply_geometric_priors=not args.no_geo_priors,
        wall_max_abs_nz=args.wall_max_abs_nz,
        horiz_min_abs_nz=args.horiz_min_abs_nz,
        floor_quantile=args.floor_quantile,
        ceiling_quantile=args.ceiling_quantile,
        normal_k=args.normal_k,
    )
