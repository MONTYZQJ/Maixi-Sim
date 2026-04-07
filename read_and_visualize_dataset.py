#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

TARGET_CONFIG = {
    "displacement": {
        "subset_dir": "DataSet_displacement",
        "result_file": "ResultFileData_displacement.npz",
        "field_key": "displacement",
        "components": ["ux", "uy", "uz", "magnitude"],
    },
    "stress": {
        "subset_dir": "DataSet_stress",
        "result_file": "ResultFileData_stress.npz",
        "field_key": "stress",
        "components": ["sigma_xx", "sigma_yy", "sigma_zz", "tau_xy", "tau_yz", "tau_xz", "sigma_eq"],
    },
}


def parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {raw}. Use true/false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read and visualize CXX dataset samples")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/afs/jiajiwei-folder/public_datasets/data_cxx"),
        help="Dataset root directory",
    )
    parser.add_argument("--target", choices=["displacement", "stress"], default="displacement")
    parser.add_argument("--sample-key", default="Sample_0", help="Sample key in NPZ, e.g. Sample_0")
    parser.add_argument("--time-index", type=int, default=10, help="Frame index along the time dimension")
    parser.add_argument(
        "--component",
        default="magnitude",
        help="Component name or integer channel index; displacement: ux/uy/uz/magnitude, stress: sigma_xx/.../sigma_eq",
    )
    parser.add_argument("--max-points", type=int, default=-1, help="Max number of points used for plotting")
    parser.add_argument("--max-edges", type=int, default=-1, help="Max number of edges used for plotting; <=0 means all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vis-mode", choices=["points", "edges", "both"], default="both")
    parser.add_argument(
        "--hotspot-top-percent",
        type=float,
        default=0.0,
        help="Stress hotspot mode: keep only top N%% high-stress points (recommended 1~5). 0 disables.",
    )
    parser.add_argument("--deformation-scale", type=float, default=1.0, help="Scale factor applied to displacement when building deformed coordinates")
    parser.add_argument("--deformed-opacity", type=float, default=1.0, help="Opacity of deformed/current geometry layer in rendered figure")
    parser.add_argument("--hide-reference", action="store_true", help="Hide undeformed reference geometry overlay")
    parser.add_argument(
        "--show-reference-field",
        type=parse_bool,
        default=1,
        help="Show reference-frame field values on undeformed geometry (true/false)",
    )
    parser.add_argument("--reference-time-index", type=int, default=0, help="Reference frame index used when --show-reference-field is enabled")
    parser.add_argument("--reference-opacity", type=float, default=0.5, help="Opacity of undeformed reference overlay")
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--line-width", type=float, default=1.2)
    parser.add_argument("--edge-opacity", type=float, default=0.35)
    parser.add_argument("--cmap", default="turbo")
    parser.add_argument("--coord-unit", default="mm", help="Coordinate unit label shown on axes, e.g. mm or m")
    parser.add_argument("--window-size", type=int, nargs=2, default=[1600, 1000], help="Render window size: width height")
    parser.add_argument("--no-grid", action="store_true", help="Disable 3D grid/axes")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/afs/jiajiwei-folder/wangjt/work/pointop/outputs"),
        help="Output directory for figure files",
    )
    parser.add_argument("--show", action="store_true", help="Display figure window")
    return parser.parse_args()


def resolve_paths(data_root: Path, target: str) -> tuple[Path, Path, str, list[str]]:
    cfg = TARGET_CONFIG[target]
    subset = data_root / cfg["subset_dir"]
    solver_path = subset / "SolverFileData.npz"
    result_path = subset / cfg["result_file"]
    if not solver_path.exists():
        raise FileNotFoundError(f"Missing solver NPZ: {solver_path}")
    if not result_path.exists():
        raise FileNotFoundError(f"Missing result NPZ: {result_path}")
    return solver_path, result_path, cfg["field_key"], cfg["components"]


def choose_component(component: str, names: list[str], values: np.ndarray) -> tuple[np.ndarray, str]:
    lowered_names = [name.lower() for name in names]
    if component.isdigit():
        index = int(component)
    else:
        component_lower = component.lower()
        if component_lower == "magnitude" and "magnitude" in lowered_names:
            return np.linalg.norm(values, axis=-1), "magnitude"
        if component_lower not in lowered_names:
            raise ValueError(f"Unknown component '{component}'. Valid options: {names} or channel index.")
        index = lowered_names.index(component_lower)

    if index < 0 or index >= values.shape[-1]:
        raise ValueError(f"Component index {index} out of range for values with {values.shape[-1]} channels")
    return values[:, index], names[index]


def summarize_sample(solver_item: dict, field_values: np.ndarray, sample_key: str, target: str) -> None:
    xyzd = np.asarray(solver_item["xyzd"])
    time_steps = np.asarray(solver_item["time"])
    edge = np.asarray(solver_item.get("edge", np.zeros((0, 2), dtype=np.int64)))

    print("=" * 80)
    print(f"Target: {target}")
    print(f"Sample key: {sample_key}")
    print(f"xyzd shape: {xyzd.shape}, dtype: {xyzd.dtype}")
    print(f"time shape: {time_steps.shape}, dtype: {time_steps.dtype}")
    print(f"edge shape: {edge.shape}, dtype: {edge.dtype}")
    print(f"field shape: {field_values.shape}, dtype: {field_values.dtype}")
    print("xyzd[:, :3] are xyz coordinates")
    print("xyzd[:, 3:] are per-sample global features replicated on all points")
    print("=" * 80)


def build_edge_polydata(coords: np.ndarray, point_scalar: np.ndarray, edge_array: np.ndarray) -> pv.PolyData:
    edges = np.asarray(edge_array, dtype=np.int64)
    if edges.size == 0:
        raise ValueError("This sample has no edge data, cannot render edges mode")

    poly = pv.PolyData(coords.astype(np.float32))
    lines = np.empty((edges.shape[0], 3), dtype=np.int64)
    lines[:, 0] = 2
    lines[:, 1:] = edges
    poly.lines = lines.reshape(-1)
    poly.point_data["field"] = point_scalar.astype(np.float32)
    return poly


def select_point_indices(num_points: int, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points > 0 and num_points > max_points:
        return rng.choice(num_points, size=max_points, replace=False)
    return np.arange(num_points)


def select_edge_array(edge_array: np.ndarray, max_edges: int, rng: np.random.Generator) -> np.ndarray:
    edges = np.asarray(edge_array, dtype=np.int64)
    if max_edges > 0 and edges.shape[0] > max_edges:
        edge_idx = rng.choice(edges.shape[0], size=max_edges, replace=False)
        return edges[edge_idx]
    return edges


def add_scene(
    plotter: pv.Plotter,
    args: argparse.Namespace,
    coords: np.ndarray,
    scalar: np.ndarray,
    edge_array: np.ndarray,
    point_indices: np.ndarray,
    title: str,
    color_limits: tuple[float, float],
    scalar_name: str,
    hotspot_threshold: float | None,
) -> None:
    scalar_bar_args = {
        "title": scalar_name,
        "width": 0.03,
        "height": 0.42,
        "position_x": 0.94,
        "position_y": 0.08,
        "label_font_size": 9,
        "title_font_size": 10,
        "n_labels": 4,
        "vertical": True,
    }
    show_scalar_bar = True
    hotspot_mode = args.target == "stress" and hotspot_threshold is not None

    if args.vis_mode in ("points", "both"):
        p_coords = coords[point_indices]
        p_scalar = scalar[point_indices]
        if hotspot_mode:
            # Deformed geometry baseline in gray; hotspots are overlaid in color.
            base_poly = pv.PolyData(p_coords.astype(np.float32))
            plotter.add_points(
                base_poly,
                color="lightgray",
                point_size=args.point_size,
                render_points_as_spheres=True,
                opacity=args.deformed_opacity,
            )
            hotspot_mask = p_scalar >= hotspot_threshold
            if np.any(hotspot_mask):
                hot_coords = p_coords[hotspot_mask]
                hot_scalar = p_scalar[hotspot_mask]
                hot_poly = pv.PolyData(hot_coords.astype(np.float32))
                hot_poly.point_data["field"] = hot_scalar.astype(np.float32)
                plotter.add_points(
                    hot_poly,
                    scalars="field",
                    cmap=args.cmap,
                    clim=color_limits,
                    point_size=args.point_size,
                    render_points_as_spheres=True,
                    opacity=args.deformed_opacity,
                    show_scalar_bar=show_scalar_bar,
                    scalar_bar_args=scalar_bar_args,
                )
                show_scalar_bar = False
        else:
            points_poly = pv.PolyData(p_coords.astype(np.float32))
            points_poly.point_data["field"] = p_scalar.astype(np.float32)
            plotter.add_points(
                points_poly,
                scalars="field",
                cmap=args.cmap,
                clim=color_limits,
                point_size=args.point_size,
                render_points_as_spheres=True,
                opacity=args.deformed_opacity,
                show_scalar_bar=show_scalar_bar,
                scalar_bar_args=scalar_bar_args,
            )
            show_scalar_bar = False

    if args.vis_mode in ("edges", "both"):
        edge_poly = build_edge_polydata(coords, scalar, edge_array)
        if hotspot_mode:
            plotter.add_mesh(
                edge_poly,
                color="lightgray",
                line_width=args.line_width,
                opacity=args.edge_opacity * args.deformed_opacity,
                render_lines_as_tubes=True,
            )
        else:
            plotter.add_mesh(
                edge_poly,
                scalars="field",
                cmap=args.cmap,
                clim=color_limits,
                line_width=args.line_width,
                opacity=args.edge_opacity * args.deformed_opacity,
                render_lines_as_tubes=True,
                show_scalar_bar=show_scalar_bar,
                scalar_bar_args=scalar_bar_args,
            )

    plotter.add_text(title, position="upper_left", font_size=11)
    if not args.no_grid:
        plotter.show_grid(
            xtitle=f"x [{args.coord_unit}]",
            ytitle=f"y [{args.coord_unit}]",
            ztitle=f"z [{args.coord_unit}]",
            font_size=10,
            n_xlabels=4,
            n_ylabels=4,
            n_zlabels=4,
            location="outer",
            ticks="outside",
        )
    plotter.add_axes()


def render_with_pyvista(
    args: argparse.Namespace,
    coords: np.ndarray,
    scalar: np.ndarray,
    edge_array: np.ndarray,
    ref_coords: np.ndarray | None,
    ref_scalar: np.ndarray | None,
    color_limits: tuple[float, float],
    frame_index: int,
    scalar_name: str,
    hotspot_threshold: float | None,
    out_path: Path,
) -> None:
    rng = np.random.default_rng(args.seed)
    plotter = pv.Plotter(off_screen=not args.show, window_size=tuple(args.window_size))
    point_indices = select_point_indices(coords.shape[0], args.max_points, rng)
    edge_subset = select_edge_array(edge_array, args.max_edges, rng)

    if ref_coords is not None and ref_scalar is not None:
        if args.vis_mode in ("points", "both"):
            ref_p_coords = ref_coords[point_indices]
            ref_points_poly = pv.PolyData(ref_p_coords.astype(np.float32))
            if args.show_reference_field:
                ref_p_scalar = ref_scalar[point_indices]
                ref_points_poly.point_data["field"] = ref_p_scalar.astype(np.float32)
                plotter.add_points(
                    ref_points_poly,
                    scalars="field",
                    cmap=args.cmap,
                    clim=color_limits,
                    point_size=max(args.point_size - 1.0, 1.0),
                    render_points_as_spheres=True,
                    opacity=args.reference_opacity,
                    show_scalar_bar=False,
                )
            else:
                plotter.add_points(
                    ref_points_poly,
                    color="lightgray",
                    point_size=max(args.point_size - 1.0, 1.0),
                    render_points_as_spheres=True,
                    opacity=args.reference_opacity,
                )

        if args.vis_mode in ("edges", "both"):
            ref_edge_poly = build_edge_polydata(ref_coords, ref_scalar, edge_subset)
            if args.show_reference_field:
                plotter.add_mesh(
                    ref_edge_poly,
                    scalars="field",
                    cmap=args.cmap,
                    clim=color_limits,
                    line_width=max(args.line_width - 0.2, 0.4),
                    opacity=args.reference_opacity,
                    render_lines_as_tubes=True,
                    show_scalar_bar=False,
                )
            else:
                plotter.add_mesh(
                    ref_edge_poly,
                    color="lightgray",
                    line_width=max(args.line_width - 0.2, 0.4),
                    opacity=args.reference_opacity,
                    render_lines_as_tubes=True,
                )

    add_scene(
        plotter,
        args,
        coords,
        scalar,
        edge_subset,
        point_indices,
        title=f"Frame t={frame_index}",
        color_limits=color_limits,
        scalar_name=scalar_name,
        hotspot_threshold=hotspot_threshold,
    )

    if args.show:
        plotter.show(screenshot=str(out_path))
    else:
        plotter.screenshot(str(out_path))
        plotter.close()


def compute_displacement_scalar_and_vector(
    component_name: str,
    displacement_frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    disp_vec = displacement_frame[:, :3]
    if component_name == "magnitude":
        scalar = np.linalg.norm(disp_vec, axis=-1)
        return scalar, disp_vec

    axis_map = {"ux": 0, "uy": 1, "uz": 2}
    if component_name in axis_map:
        axis = axis_map[component_name]
        single_axis_disp = np.zeros_like(disp_vec)
        single_axis_disp[:, axis] = disp_vec[:, axis]
        scalar = np.abs(disp_vec[:, axis])
        return scalar, single_axis_disp

    # Fallback: keep full displacement geometry and use provided scalar semantics.
    scalar = np.linalg.norm(disp_vec, axis=-1)
    return scalar, disp_vec


def compute_sample_color_limits(
    target: str,
    component_name: str,
    field_all: np.ndarray,
    component_names: list[str],
) -> tuple[float, float]:
    min_value = np.inf
    max_value = -np.inf

    for frame in field_all:
        if target == "displacement":
            frame_scalar, _ = compute_displacement_scalar_and_vector(component_name, frame)
        else:
            frame_scalar, _ = choose_component(component_name, component_names, frame)
        min_value = min(min_value, float(np.min(frame_scalar)))
        max_value = max(max_value, float(np.max(frame_scalar)))

    if not np.isfinite(min_value) or not np.isfinite(max_value):
        return 0.0, 1.0
    if max_value <= min_value:
        eps = max(abs(max_value) * 1e-6, 1e-6)
        return min_value - eps, max_value + eps
    return min_value, max_value


def main() -> None:
    args = parse_args()
    if args.hotspot_top_percent < 0 or args.hotspot_top_percent >= 100:
        raise ValueError("--hotspot-top-percent must be in [0, 100)")
    if args.deformed_opacity < 0 or args.deformed_opacity > 1:
        raise ValueError("--deformed-opacity must be in [0, 1]")

    solver_path, result_path, field_key, component_names = resolve_paths(args.data_root, args.target)
    solver_npz = np.load(solver_path, allow_pickle=True)
    result_npz = np.load(result_path, allow_pickle=True)

    if args.sample_key not in solver_npz.files:
        raise KeyError(
            f"Sample key '{args.sample_key}' not found in {solver_path.name}. Available keys include: {solver_npz.files[:5]}"
        )

    solver_item = solver_npz[args.sample_key].item()
    result_item = result_npz[args.sample_key].item()

    xyzd = np.asarray(solver_item["xyzd"], dtype=np.float64)
    coords = xyzd[:, :3]
    time_steps = np.asarray(solver_item["time"])
    field_all = np.asarray(result_item[field_key], dtype=np.float64)

    summarize_sample(solver_item, field_all, args.sample_key, args.target)

    if args.time_index < 0 or args.time_index >= field_all.shape[0]:
        raise IndexError(f"time-index {args.time_index} out of range [0, {field_all.shape[0] - 1}]")
    if args.reference_time_index < 0 or args.reference_time_index >= field_all.shape[0]:
        raise IndexError(f"reference-time-index {args.reference_time_index} out of range [0, {field_all.shape[0] - 1}]")

    field_t = field_all[args.time_index]
    field_comp, comp_name = choose_component(args.component, component_names, field_t)

    edge_array = np.asarray(solver_item.get("edge", np.zeros((0, 2), dtype=np.int64)), dtype=np.int64)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_coords = coords
    plot_scalar = field_comp
    ref_overlay_coords: np.ndarray | None = None
    ref_overlay_scalar: np.ndarray | None = None

    if args.target == "displacement":
        if args.hotspot_top_percent > 0:
            raise ValueError("--hotspot-top-percent is only supported for target=stress")
        disp_scalar, disp_vector = compute_displacement_scalar_and_vector(comp_name, field_t)
        plot_coords = coords + args.deformation_scale * disp_vector
        plot_scalar = disp_scalar
    else:
        disp_result_path = args.data_root / TARGET_CONFIG["displacement"]["subset_dir"] / TARGET_CONFIG["displacement"]["result_file"]
        disp_npz = np.load(disp_result_path, allow_pickle=True)
        if args.sample_key not in disp_npz.files:
            raise KeyError(f"Sample key '{args.sample_key}' missing in displacement file: {disp_result_path}")
        disp_frame = np.asarray(disp_npz[args.sample_key].item()["displacement"], dtype=np.float64)[args.time_index]
        plot_coords = coords + args.deformation_scale * disp_frame[:, :3]
        # Stress keeps its own requested scalar for coloring.
        plot_scalar = field_comp

    hotspot_threshold: float | None = None
    if args.target == "stress" and args.hotspot_top_percent > 0:
        hotspot_threshold = float(np.quantile(plot_scalar, 1.0 - args.hotspot_top_percent / 100.0))

    # Reference overlay is only enabled when explicitly requested.
    if not args.hide_reference and args.show_reference_field:
        ref_overlay_coords = coords
        reference_frame = field_all[args.reference_time_index]
        if args.target == "displacement":
            ref_overlay_scalar, _ = compute_displacement_scalar_and_vector(comp_name, reference_frame)
        else:
            ref_overlay_scalar, _ = choose_component(comp_name, component_names, reference_frame)

    color_limits = compute_sample_color_limits(args.target, comp_name, field_all, component_names)

    if hotspot_threshold is not None:
        kept = int(np.sum(plot_scalar >= hotspot_threshold))
        print(
            f"Hotspot mode enabled: top {args.hotspot_top_percent:.2f}% | threshold={hotspot_threshold:.6g} | kept_points={kept}/{plot_scalar.shape[0]}"
        )

    out_name = f"{args.target}_{args.sample_key}_t{args.time_index:03d}_{comp_name}_{args.vis_mode}.png"
    out_path = args.output_dir / out_name
    render_with_pyvista(
        args,
        plot_coords,
        plot_scalar,
        edge_array,
        ref_overlay_coords,
        ref_overlay_scalar,
        color_limits,
        args.time_index,
        comp_name,
        hotspot_threshold,
        out_path,
    )

    print(
        f"Rendered mode={args.vis_mode}, points={coords.shape[0]}, edges={edge_array.shape[0]}, "
        f"time={float(time_steps[args.time_index]):.6f}, component={comp_name}"
    )
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
