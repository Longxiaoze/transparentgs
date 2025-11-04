import argparse
import os
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from gaussian_renderer import GaussianModel
from renderer import render_gaussian_to_file

RendererLike = Union[str, Callable]
GaussianLike = Union[GaussianModel, str, os.PathLike]
PoseLike = Union[np.ndarray, torch.Tensor, Sequence[float], str, os.PathLike]
CenterLike = Union[Sequence[float], np.ndarray, torch.Tensor, None]

RENDERER_ALIASES = {
    "pinhole": "simple_render",
    "panorama": "simple_render_panorama",
    "fisheye": "simple_render_fisheye",
}


def _ensure_gaussian(gaussian: GaussianLike) -> GaussianModel:
    if isinstance(gaussian, GaussianModel):
        return gaussian
    model = GaussianModel(3)
    model.load_ply(str(gaussian))
    return model


def _parse_numeric_sequence(seq: Union[str, Iterable[float]]) -> np.ndarray:
    if isinstance(seq, str):
        tokens = seq.replace(",", " ").split()
        values = [float(token) for token in tokens]
    else:
        values = [float(token) for token in seq]
    return np.asarray(values, dtype=np.float32)


def _ensure_pose(pose: PoseLike) -> np.ndarray:
    if isinstance(pose, (np.ndarray, torch.Tensor)):
        arr = pose.detach().cpu().numpy() if isinstance(pose, torch.Tensor) else pose
        arr = np.asarray(arr, dtype=np.float32)
    elif isinstance(pose, (list, tuple)):
        arr = _parse_numeric_sequence(pose)
    else:
        pose_path = os.fspath(pose)
        if pose_path.endswith(".npy"):
            arr = np.load(pose_path).astype(np.float32)
        else:
            try:
                arr = np.loadtxt(pose_path, dtype=np.float32)
            except (OSError, ValueError):
                arr = _parse_numeric_sequence(pose_path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size != 16:
        raise ValueError("pose must contain 16 values forming a 4x4 matrix.")
    arr = arr.reshape(4, 4)
    return arr


def _ensure_camera_center(center: CenterLike) -> Optional[np.ndarray]:
    if center is None:
        return None
    if isinstance(center, (np.ndarray, torch.Tensor)):
        arr = center.detach().cpu().numpy() if isinstance(center, torch.Tensor) else center
        arr = np.asarray(arr, dtype=np.float32)
    else:
        arr = _parse_numeric_sequence(center)
    if arr.size != 3:
        raise ValueError("camera center must contain exactly 3 values.")
    return arr.reshape(3)


def _ensure_bg_color(color: Union[str, Sequence[float]]) -> Tuple[float, float, float]:
    arr = _parse_numeric_sequence(color)
    if arr.size != 3:
        raise ValueError("bg_color must contain exactly 3 values.")
    return tuple(float(v) for v in arr)


def render_gaussian_image(
    gaussian: GaussianLike,
    pose: PoseLike,
    output_path: str,
    *,
    renderer: RendererLike = "simple_render",
    height: int = 512,
    width: int = 512,
    fovx: float = 60.0,
    fovy: float = 60.0,
    camera_center: CenterLike = None,
    bg_color: Union[str, Sequence[float]] = (0.0, 0.0, 0.0),
    scale_modifier: float = 1.0,
    proxy: Optional[GaussianModel] = None,
    pc_label: int = 0,
    proxy_label: int = 1,
    to_uint8: bool = True,
) -> np.ndarray:
    """Render a GaussianModel view and save it to disk."""
    gaussian_model = _ensure_gaussian(gaussian)
    pose_matrix = _ensure_pose(pose)
    camera_center_vec = _ensure_camera_center(camera_center)
    bg = _ensure_bg_color(bg_color)

    renderer_key = renderer
    if isinstance(renderer, str):
        renderer_key = RENDERER_ALIASES.get(renderer, renderer)

    return render_gaussian_to_file(
        gaussian_model,
        pose_matrix,
        output_path,
        renderer=renderer_key,
        height=height,
        width=width,
        fovx=fovx,
        fovy=fovy,
        camera_center=camera_center_vec,
        bg_color=bg,
        scale_modifier=scale_modifier,
        proxy=proxy,
        pc_label=pc_label,
        proxy_label=proxy_label,
        to_uint8=to_uint8,
    )


def _cli():
    parser = argparse.ArgumentParser(description="Render a Gaussian Splatting model to a still image.")
    parser.add_argument("--gaussian", required=True, help="Path to a GaussianModel .ply file.")
    parser.add_argument("--pose", required=True,
                        help="Path to a 4x4 pose (txt/npy) or 16 numbers separated by space/comma.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--renderer", default="pinhole",
                        choices=sorted(set(RENDERER_ALIASES) | {"simple_render", "simple_render_panorama", "simple_render_fisheye"}),
                        help="Rendering function to use.")
    parser.add_argument("--height", type=int, default=512, help="Output image height.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument("--fovx", type=float, default=60.0, help="Horizontal field of view in degrees.")
    parser.add_argument("--fovy", type=float, default=60.0, help="Vertical field of view in degrees.")
    parser.add_argument("--camera-center", default=None,
                        help="Optional camera position override (file path or 3 numbers).")
    parser.add_argument("--bg-color", default="0,0,0", help="Background color RGB in [0,1], e.g. '0,0,0'.")
    parser.add_argument("--scale-modifier", type=float, default=1.0, help="Scale modifier forwarded to the renderer.")
    parser.add_argument("--float-output", action="store_true",
                        help="Keep floating point pixel values instead of converting to uint8.")
    args = parser.parse_args()

    render_gaussian_image(
        gaussian=args.gaussian,
        pose=args.pose,
        output_path=args.output,
        renderer=args.renderer,
        height=args.height,
        width=args.width,
        fovx=args.fovx,
        fovy=args.fovy,
        camera_center=args.camera_center,
        bg_color=args.bg_color,
        scale_modifier=args.scale_modifier,
        to_uint8=not args.float_output,
    )


if __name__ == "__main__":
    _cli()
# example usage:
# python render_gaussian_image.py --gaussian model.ply --pose identity_4x4.npy --output rendered_image.png --renderer pinhole --height 512 --width 512 --fovx 60 --fovy 60 --bg-color 0,0,0
# python render_gaussian_image.py --gaussian model.ply --pose identity_4x4.npy --output rendered_image.png --renderer fisheye --height 512 --width 512 --fovx 60 --fovy 60 --bg-color 0,0,0
# python render_gaussian_image.py --gaussian model.ply --pose identity_4x4.npy --output rendered_image.png --renderer panorama --height 512 --width 1024--bg-color 0,0,0
