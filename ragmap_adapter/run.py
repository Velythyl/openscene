"""ragmap-run: OpenScene's multi-view 2D fused OpenSeg features behind RAGMAP's object-mapping contract.

    ragmap-run --input /input --output /output [key=value ...]

INPUT: RAGMAP's shared contract (``ragmap/objects/base.py``)::

    meta.json            {"fx","fy","cx","cy","width","height","depth_scale",...}
    frames.jsonl         {"frame_index","observation_id","rgb","depth","pose"} per frame
    rgb/, depth/         hard links to the dataset's own files
    scene_cloud.ply      the dataset's own scene geometry, *in the dataset's frame*
                         (a hard link of NICE-SLAM's <scene>_mesh.ply or the walk's scene_rgb.ply)
    dataset_frame.json   {"poses": {frame_index: 4x4}}: each frame's OpenCV camera-to-world
                         in that same frame (the dataset's own pose files), written by the module

OpenLex3D (Supp. G): "OpenScene was executed at intervals of 10 frames. For evaluation,
we use the multi-view 2D fused features". What runs, upstream code unmodified
(``pengsongyou/openscene`` @ 0f369bc):

1. 3D input: ``scripts/preprocess/preprocess_3d_replica.py``'s ``process_one_scene``
   (re-stated here, since the script runs its whole scene list at import): the ply's
   vertices via plyfile, ``coords = v[:, :3]``, ``colors = v[:, -3:] / 127.5 - 1``,
   labels 255, saved as ``<scene>.pth``.
2. 2D input: ``scripts/preprocess/preprocess_2d_replica.py``'s ``process_one_scene``
   (re-stated): every ``stride``-th frame (``files[::10]``), RGB and depth resized to
   ``img_dim`` with cv2 INTER_LINEAR, written as ``color/<i>.jpg`` / ``depth/<i>.png``
   with imageio, the pose as ``pose/<i>.txt``, and the intrinsics adjusted by
   ``adjust_intrinsic`` into ``intrinsics.txt``.
3. Fusion: ``scripts/feature_fusion/replica_openseg.py``'s ``process_one_scene``, called
   with the arguments its ``main()`` sets (img_dim, depth_scale, visibility_threshold
   0.25, cut_num_pixel_boundary 10, feat_dim 768, n_split_points 2 000 000,
   num_rand_file_per_scene 1, seed 1457), OpenSeg loaded as ``main()`` loads it.
4. Export, read off the published OpenLex3D files (the authors released no script for
   this step): every input point gets the voxel of its OpenScene ``sparse_quantize``
   representative (``dataset/voxelization_utils.py``, quantization 0.05, FNV hash, on
   the *unshifted* coordinates) and that representative's fused feature (zero when no
   view saw it), as in ``run/evaluate.py``'s fusion branch (``feat_3d[vox_ind]`` then
   ``[inds_reverse]``). Written as ``<scene>_coords.npy`` (voxel units),
   ``<scene>_feat_3d.npy`` (fp16), ``<scene>_fusion.ply`` and ``<scene>_input.ply``
   (``util.util.export_pointcloud``'s Open3D write, input colours ``(c + 1) / 2``), then
5. the OpenLex3D authors' converter, ``openlex3d @ 8f4efb0
   openlex3d/scripts/preprocessing/openscene_to_openlex_format.py`` (run unmodified),
   which writes ``embeddings.npy``, ``index.npy`` (its ``np.linspace(0, n, n)``) and the
   metric ``input.ply`` (voxel units x 0.05).

OUTPUT::

    openscene/input.ply, embeddings.npy, index.npy   step 5's files, in the dataset frame
    objects.jsonl                                   empty: a dense map has no objects
    run.json

Every exception is recorded in run.json (``status: failed``) before exiting 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("OPENSCENE_ROOT", "/opt/openscene"))
FUSION_DIR = ROOT / "scripts" / "feature_fusion"
OPENLEX3D_ROOT = Path(os.environ.get("OPENLEX3D_ROOT", "/opt/openlex3d"))
CONVERTER = OPENLEX3D_ROOT / "openlex3d" / "scripts" / "preprocessing" / "openscene_to_openlex_format.py"
WEIGHTS = Path(os.environ.get("OPENSCENE_WEIGHTS", "/weights"))
SCHEMA = "ragmap.openscene/v1"

DEFAULTS = {
    # preprocess_2d_replica.py: sample_freq = 10 ("intervals of 10 frames").
    "openscene.stride": "10",
    # replica_openseg.py main(): img_dim (640, 360) (= preprocess_2d_replica.py's).
    "openscene.img_dim": "640x360",
    # replica_openseg.py main(): visibility_threshold 0.25, cut_num_pixel_boundary 10.
    "openscene.visibility_threshold": "0.25",
    "openscene.cut_bound": "10",
    # replica_openseg.py main(): n_split_points 2 000 000, num_rand_file_per_scene 1.
    "openscene.n_split_points": "2000000",
    # The voxel of the export (step 4); the converter's `scale = 0.05`.
    "openscene.voxel_size": "0.05",
    "ragmap.openseg_model": "openseg_colab/exported_model",
    # Keep the preprocessed 2D/3D data and the fused .pt (large).
    "ragmap.keep_intermediate": "false",
}


def _bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _dim(text: str) -> tuple[int, int]:
    width, sep, height = str(text).lower().partition("x")
    if not sep:
        raise SystemExit(f"openscene.img_dim {text!r} is not <width>x<height>")
    return int(width), int(height)


def parse_overrides(items) -> dict:
    config = dict(DEFAULTS)
    for item in items:
        if "=" not in item:
            raise SystemExit(f"override {item!r} is not key=value")
        key, value = item.split("=", 1)
        if key not in config:
            raise SystemExit(f"unknown override {key!r} (known: {sorted(config)})")
        config[key] = value
    return config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ragmap-run", description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", default="/input", type=Path)
    parser.add_argument("--output", default="/output", type=Path)
    parser.add_argument("overrides", nargs="*", help="openscene.<k>=v | ragmap.<k>=v")
    args = parser.parse_args(argv)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    run: dict = {"schema": SCHEMA, "status": "running", "openscene_git_sha": os.environ.get("OPENSCENE_GIT_SHA"),
                 "openlex3d_sha": os.environ.get("OPENLEX3D_SHA"), "timings": {}}
    started = time.time()
    try:
        _main(args, run)
        run["status"] = "ok"
        return_code = 0
    except BaseException as error:  # noqa: BLE001 - recorded, then re-signalled by the exit code
        run["status"] = "failed"
        run["error"] = f"{type(error).__name__}: {error}"
        run["traceback"] = traceback.format_exc()
        print(run["traceback"], file=sys.stderr, flush=True)
        return_code = 1
    run["timings"]["total_s"] = round(time.time() - started, 2)
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    return return_code


def prepare_3d(cloud: Path, out_pth: Path) -> dict:
    """preprocess_3d_replica.py process_one_scene, for one ply."""

    import plyfile
    import torch

    a = plyfile.PlyData().read(str(cloud))
    v = np.array([list(x) for x in a.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])
    colors = np.ascontiguousarray(v[:, -3:]) / 127.5 - 1
    # no GT labels are provided, set all to 255
    labels = 255 * np.ones((coords.shape[0],), dtype=np.int32)
    out_pth.parent.mkdir(parents=True, exist_ok=True)
    torch.save((coords, colors, labels), str(out_pth))
    names = [p.name for p in a.elements[0].properties]
    return {"points": int(len(coords)), "coords_dtype": str(coords.dtype), "properties": names,
            "color_properties": names[-3:]}


def prepare_2d(input_dir: Path, frames: list[dict], poses: dict, meta: dict, stride: int,
               img_dim: tuple[int, int], out_dir: Path) -> dict:
    """preprocess_2d_replica.py, for one scene: every `stride`-th frame, resized to img_dim."""

    import cv2
    import imageio

    sys.path.insert(0, str(FUSION_DIR))
    from fusion_util import adjust_intrinsic, make_intrinsic

    original = (int(meta["width"]), int(meta["height"]))
    intrinsics = make_intrinsic(fx=float(meta["fx"]), fy=float(meta["fy"]), mx=float(meta["cx"]), my=float(meta["cy"]))
    intrinsics = adjust_intrinsic(intrinsics, original, img_dim)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "intrinsics.txt", intrinsics)
    scene_dir = out_dir / "scene"
    for sub in ("color", "depth", "pose"):
        (scene_dir / sub).mkdir(parents=True, exist_ok=True)
    chosen = frames[::stride]
    for img_id, frame in enumerate(chosen):
        img = imageio.v3.imread(str(input_dir / frame["rgb"]))
        img = cv2.resize(img, img_dim, interpolation=cv2.INTER_LINEAR)
        imageio.imwrite(str(scene_dir / "color" / f"{img_id}.jpg"), img)
        depth = imageio.v3.imread(str(input_dir / frame["depth"])).astype(np.uint16)
        depth = cv2.resize(depth, img_dim, interpolation=cv2.INTER_LINEAR)
        imageio.imwrite(str(scene_dir / "depth" / f"{img_id}.png"), depth)
        np.savetxt(scene_dir / "pose" / f"{img_id}.txt", np.asarray(poses[str(frame["frame_index"])], dtype=np.float64))
    return {"frames_in": len(frames), "frames_fused": len(chosen), "stride": stride, "img_dim": list(img_dim),
            "original_dim": list(original), "intrinsics": intrinsics.tolist(),
            "first": Path(chosen[0]["rgb"]).name, "last": Path(chosen[-1]["rgb"]).name}


def fuse(data_2d: Path, pth: Path, fused_dir: Path, config: dict, depth_scale: float, img_dim, run: dict) -> Path:
    """replica_openseg.py main() for one scene, then its process_one_scene."""

    import torch
    import tensorflow as tf2
    import tensorflow.compat.v1 as tf

    sys.path.insert(0, str(FUSION_DIR))
    import replica_openseg
    from fusion_util import PointCloudToImageMapper

    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    model_dir = WEIGHTS / config["ragmap.openseg_model"]
    if not (model_dir / "saved_model.pb").is_file():
        raise RuntimeError(f"{model_dir}/saved_model.pb is missing (mount the weights directory at {WEIGHTS})")
    gpus = tf2.config.list_physical_devices("GPU")
    run["tensorflow_gpus"] = [g.name for g in gpus]
    if not gpus:
        raise RuntimeError("TensorFlow sees no GPU: OpenSeg would run on the CPU")
    args = argparse.Namespace()
    args.scene = "scene"
    args.depth_scale = depth_scale
    args.cut_num_pixel_boundary = int(config["openscene.cut_bound"])
    args.keep_features_in_memory = False
    args.feat_dim = 768
    args.data_root_2d = str(data_2d)
    args.n_split_points = int(config["openscene.n_split_points"])
    args.num_rand_file_per_scene = 1
    args.openseg_model = tf2.saved_model.load(str(model_dir), tags=[tf.saved_model.tag_constants.SERVING],)
    args.text_emb = tf.zeros([1, 1, args.feat_dim])
    intrinsics = np.loadtxt(os.path.join(args.data_root_2d, "intrinsics.txt"))
    args.point2img_mapper = PointCloudToImageMapper(
        image_dim=img_dim, intrinsics=intrinsics,
        visibility_threshold=float(config["openscene.visibility_threshold"]),
        cut_bound=args.cut_num_pixel_boundary)
    fused_dir.mkdir(parents=True, exist_ok=True)
    replica_openseg.process_one_scene(str(pth), str(fused_dir), args)
    out = fused_dir / "scene_0.pt"
    if not out.is_file():
        raise RuntimeError(f"fusion wrote no {out}")
    return out


def export(pth: Path, fused: Path, scene_dir: Path, voxel: float) -> dict:
    """Step 4: per-point voxel coordinates and features of the sparse_quantize representative."""

    import torch
    import open3d as o3d

    sys.path.insert(0, str(ROOT))
    from dataset.voxelization_utils import sparse_quantize

    locs_in, feats_in, _ = torch.load(str(pth))
    processed = torch.load(str(fused))
    feat, mask_full = processed["feat"], processed["mask_full"]
    if isinstance(mask_full, np.ndarray):
        mask_full = torch.from_numpy(mask_full)
    # FusedFeatureLoader, val/test split: every point, zero where no feature was fused.
    feat_3d = torch.zeros((locs_in.shape[0], feat.shape[1]), dtype=feat.dtype)
    feat_3d[mask_full] = feat
    inds, inds_reverse = sparse_quantize(locs_in, return_index=True, quantization_size=voxel)
    discrete = np.floor(locs_in / np.array([voxel] * locs_in.shape[1]))
    coords = discrete[inds][inds_reverse]
    feats = feat_3d[inds][inds_reverse].numpy()
    scene_dir.mkdir(parents=True, exist_ok=True)
    np.save(scene_dir / "scene_coords.npy", coords)
    np.save(scene_dir / "scene_feat_3d.npy", feats)
    for name, colors in (("scene_fusion.ply", None), ("scene_input.ply", (feats_in + 1) / 2)):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(scene_dir / name), pcd)
    return {"points": int(len(coords)), "voxels": int(len(inds)), "fused_points": int(mask_full.sum()),
            "zero_feature_points": int((~feats.any(1)).sum()), "feat_dtype": str(feats.dtype)}


def _main(args, run: dict) -> None:
    input_dir: Path = args.input.resolve()
    output: Path = args.output.resolve()
    config = parse_overrides(args.overrides)
    run["config"] = config
    meta = json.loads((input_dir / "meta.json").read_text())
    frames = [json.loads(line) for line in (input_dir / "frames.jsonl").read_text().splitlines() if line.strip()]
    if not frames:
        raise RuntimeError("frames.jsonl is empty")
    frames.sort(key=lambda frame: int(frame["frame_index"]))
    cloud = input_dir / "scene_cloud.ply"
    dataset_frame = json.loads((input_dir / "dataset_frame.json").read_text())
    poses = dataset_frame["poses"]
    missing = [f["frame_index"] for f in frames if str(f["frame_index"]) not in poses]
    if missing:
        raise RuntimeError(f"dataset_frame.json has no pose for frames {missing[:5]}...")

    from importlib import metadata

    def version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    run["versions"] = {"python": sys.version.split()[0],
                       **{name: version(name) for name in ("tensorflow", "torch", "numpy", "open3d", "imageio",
                                                           "opencv-python-headless", "plyfile")}}
    model_dir = WEIGHTS / config["ragmap.openseg_model"]
    run["weights"] = {"openseg_model": str(model_dir),
                      "saved_model_pb_sha256": _sha256(model_dir / "saved_model.pb") if (model_dir / "saved_model.pb").is_file() else None}
    stride = int(config["openscene.stride"])
    img_dim = _dim(config["openscene.img_dim"])
    voxel = float(config["openscene.voxel_size"])
    run["input"] = {"frames": len(frames), "width": meta["width"], "height": meta["height"], "fx": meta["fx"],
                    "fy": meta["fy"], "cx": meta["cx"], "cy": meta["cy"], "depth_scale": meta["depth_scale"],
                    "scene_cloud_sha256": _sha256(cloud)}

    work = output / "_work"
    final = output / "openscene"
    for path in (work, final):
        if path.exists():
            shutil.rmtree(path)
    pth = work / "replica_3d" / "scene.pth"
    t0 = time.time()
    run["prepare_3d"] = prepare_3d(cloud, pth)
    run["prepare_2d"] = prepare_2d(input_dir, frames, poses, meta, stride, img_dim, work / "replica_2d")
    run["timings"]["prepare_s"] = round(time.time() - t0, 2)
    print(f"[ragmap] prepared {run['prepare_3d']['points']} points, {run['prepare_2d']['frames_fused']} frames "
          f"at {img_dim}", flush=True)
    t0 = time.time()
    fused = fuse(work / "replica_2d", pth, work / "fused", config, float(meta["depth_scale"]), img_dim, run)
    run["timings"]["fusion_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    run["export"] = export(pth, fused, final, voxel)
    # The authors' converter, unmodified: `python openscene_to_openlex_format.py <scene_path>`.
    sys.argv = [str(CONVERTER), str(final)]
    cwd = os.getcwd()
    try:
        os.chdir(final)
        runpy.run_path(str(CONVERTER), run_name="__main__")
    finally:
        os.chdir(cwd)
    for name in ("scene_coords.npy", "scene_feat_3d.npy", "scene_fusion.ply", "scene_input.ply", "fusion.ply"):
        (final / name).unlink(missing_ok=True)
    run["timings"]["export_s"] = round(time.time() - t0, 2)

    import open3d as o3d

    embeddings = np.load(final / "embeddings.npy", mmap_mode="r")
    index = np.load(final / "index.npy", mmap_mode="r")
    points = np.asarray(o3d.io.read_point_cloud(str(final / "input.ply")).points)
    if not (len(points) == len(embeddings) == len(index)):
        raise RuntimeError(f"export sizes disagree: {len(points)} points, {embeddings.shape} embeddings, {len(index)} index")
    sample = np.asarray(embeddings[:: max(1, len(embeddings) // 20000)], dtype=np.float32)
    norms = np.linalg.norm(sample, axis=1)
    run["counts"] = {
        "frames_in": len(frames), "frames_fused": run["prepare_2d"]["frames_fused"], "points": int(len(points)),
        "voxels": run["export"]["voxels"], "embedding_dim": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype), "zero_feature_points": run["export"]["zero_feature_points"],
        "embedding_norm_sample": {"min": float(norms.min()), "median": float(np.median(norms)),
                                  "max": float(norms.max()), "n": int(len(norms))},
        "bbox_min": points.min(0).tolist(), "bbox_max": points.max(0).tolist(),
    }
    (output / "objects.jsonl").write_text("", encoding="utf-8")
    if not _bool(config["ragmap.keep_intermediate"]):
        shutil.rmtree(work)
    print(f"[ragmap] OpenScene: {run['counts']['points']} points x {run['counts']['embedding_dim']} from "
          f"{run['counts']['frames_fused']} frames -> {final}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
