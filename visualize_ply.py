import argparse
from pathlib import Path

import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

ID_TO_NAME = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    7: "unassigned",
}


def load_point_cloud(file_path: Path) -> o3d.geometry.PointCloud:
    print(f"Loading point cloud: {file_path}")
    if file_path.suffix.lower() not in [".ply", ".pcd"]:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    pcd = o3d.io.read_point_cloud(str(file_path))
    print(f"Loaded {len(pcd.points)} points")
    return pcd


def create_legend_geometries(
    colors,
    origin=(0, 0, 0),
    box_size=0.2,
    gap=0.05,
):
    geometries = []
    ox, oy, oz = origin
    for idx, color in enumerate(colors):
        box = o3d.geometry.TriangleMesh.create_box(
            width=box_size,
            height=box_size,
            depth=box_size,
        )
        box.paint_uniform_color(color)
        box.translate((ox, oy + idx * (box_size + gap), oz))
        geometries.append(box)
    return geometries


def compute_legend_origin(
    pcd,
    box_size=0.2,
    gap=0.05,
    position="top-right",
):
    bounds = pcd.get_axis_aligned_bounding_box()
    min_bound = bounds.get_min_bound()
    max_bound = bounds.get_max_bound()

    x = max_bound[0] + (box_size + gap)
    y = max_bound[1] - (box_size + gap)
    z = max_bound[2]

    if position == "top-left":
        x = min_bound[0] - (box_size + gap)
        y = max_bound[1] - (box_size + gap)
    elif position == "bottom-right":
        x = max_bound[0] + (box_size + gap)
        y = min_bound[1] + (box_size + gap)
    elif position == "bottom-left":
        x = min_bound[0] - (box_size + gap)
        y = min_bound[1] + (box_size + gap)

    return (x, y, z)


def show_color_legend_window(labels, colors, title="Legend"):

    fig, ax = plt.subplots(figsize=(4, max(3, len(labels) * 0.4)))
    ax.set_axis_off()

    for idx, (label, color) in enumerate(zip(labels, colors)):
        ax.scatter([], [], c=[color], label=f"{idx}: {label}")

    ax.legend(loc="center left", frameon=False)
    fig.canvas.manager.set_window_title(title)
    plt.tight_layout()
    plt.show()


def get_instance_colors(pcd, decimals=3):
    colors = np.asarray(pcd.colors)
    if colors.size == 0:
        return np.empty((0, 3)), np.empty((0,), dtype=int)

    rounded = np.round(colors, decimals=decimals)
    unique_colors, counts = np.unique(rounded, axis=0, return_counts=True)
    return unique_colors, counts


def get_instance_legend_from_dir(instances_dir: Path):
    labels = []
    colors = []

    if not instances_dir.exists():
        return labels, colors

    for class_dir in sorted(p for p in instances_dir.iterdir() if p.is_dir()):
        class_name = class_dir.name
        instance_files = sorted(class_dir.glob("*.ply"))
        for idx, instance_file in enumerate(instance_files, start=1):
            pcd = o3d.io.read_point_cloud(str(instance_file))
            point_colors = np.asarray(pcd.colors)
            if point_colors.size == 0:
                continue
            mean_color = point_colors.mean(axis=0)
            labels.append(f"{class_name}_{idx}")
            colors.append(mean_color.tolist())

    return labels, colors


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a point cloud (.ply/.pcd)")
    parser.add_argument("--input_file", required=True, help="Path to .ply or .pcd file")
    parser.add_argument("--window_name", default="Point Cloud", help="Window title")
    parser.add_argument("--width", type=int, default=1024, help="Window width")
    parser.add_argument("--height", type=int, default=768, help="Window height")
    parser.add_argument("--legend", action="store_true", help="Show class color legend")
    parser.add_argument(
        "--legend-mode",
        choices=["instance", "class"],
        default="instance",
        help="Legend mode (instance uses point colors)",
    )
    parser.add_argument(
        "--legend-position",
        choices=["top-right", "top-left", "bottom-right", "bottom-left"],
        default="top-right",
        help="Where to place the legend relative to the point cloud bounds",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=len(ID_TO_NAME),
        help="Number of classes to show in the legend",
    )
    parser.add_argument(
        "--legend-max",
        type=int,
        default=30,
        help="Maximum legend entries (instance mode)",
    )
    parser.add_argument(
        "--legend-from-instances-dir",
        default=None,
        help="Optional directory with per-class instance PLYs for naming",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    pcd = load_point_cloud(input_path)

    legend_geometries = None
    if args.legend:
        legend_origin = compute_legend_origin(pcd, position=args.legend_position)

        if args.legend_mode == "class":
            cmap = plt.get_cmap("tab20", args.num_classes)
            colors = [cmap(i % args.num_classes)[:3] for i in range(args.num_classes)]
            labels = [ID_TO_NAME.get(i, f"class_{i}") for i in range(args.num_classes)]
            print("\nLegend colors (class_id -> name):")
            for class_id, class_name in enumerate(labels):
                print(f"  {class_id}: {class_name}")
            legend_geometries = create_legend_geometries(
                colors,
                origin=legend_origin,
            )
            show_color_legend_window(labels, colors, title="Class Legend")
        else:
            instance_dir = None
            if args.legend_from_instances_dir:
                instance_dir = Path(args.legend_from_instances_dir)
            elif input_path.name == "all_instances_combined.ply":
                instance_dir = input_path.parent

            labels, colors = ([], [])
            if instance_dir:
                labels, colors = get_instance_legend_from_dir(instance_dir)

            if labels and colors:
                max_items = min(args.legend_max, len(colors))
                labels = labels[:max_items]
                colors = colors[:max_items]
                print("\nLegend colors (instance -> label):")
                for label in labels:
                    print(f"  {label}")
                legend_geometries = create_legend_geometries(
                    colors,
                    origin=legend_origin,
                )
                show_color_legend_window(labels, colors, title="Instance Legend")
            else:
                unique_colors, counts = get_instance_colors(pcd)
                if unique_colors.size == 0:
                    print("\nLegend skipped: point cloud has no colors.")
                else:
                    order = np.argsort(-counts)
                    unique_colors = unique_colors[order]
                    counts = counts[order]
                    max_items = min(args.legend_max, len(unique_colors))
                    colors = unique_colors[:max_items]
                    labels = [f"inst_{i} (n={int(c)})" for i, c in enumerate(counts[:max_items])]
                    print("\nLegend colors (instance -> count):")
                    for i, count in enumerate(counts[:max_items]):
                        print(f"  inst_{i}: {int(count)}")
                    legend_geometries = create_legend_geometries(
                        colors,
                        origin=legend_origin,
                    )
                    show_color_legend_window(labels, colors, title="Instance Legend")

    geometries = [pcd]
    if legend_geometries:
        geometries.extend(legend_geometries)

    o3d.visualization.draw_geometries(
        geometries,
        window_name=args.window_name,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
