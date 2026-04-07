import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# Matches read_and_visualize_dataset.py: subset dir, result filename, field keys
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


def resolve_data_paths(data_root: str, target: str) -> Tuple[str, str, str, List[str]]:
    """Same layout as read_and_visualize_dataset.resolve_paths: SolverFileData + ResultFileData_*."""
    if target not in TARGET_CONFIG:
        raise ValueError(f"Unknown target={target}")
    cfg = TARGET_CONFIG[target]
    root = os.path.abspath(data_root)
    subset = os.path.join(root, cfg["subset_dir"])
    solver_path = os.path.join(subset, "SolverFileData.npz")
    result_path = os.path.join(subset, cfg["result_file"])
    if not os.path.isfile(solver_path):
        raise FileNotFoundError(f"Missing solver/geometry NPZ (expected like visualization script): {solver_path}")
    if not os.path.isfile(result_path):
        raise FileNotFoundError(f"Missing result field NPZ: {result_path}")
    return solver_path, result_path, cfg["field_key"], list(cfg["components"])


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _np_to_torch_f32(arr: np.ndarray) -> torch.Tensor:
    """
    ndarray (float32) -> Tensor without ``torch.from_numpy`` (some builds raise
    ``RuntimeError: Numpy is not available``).
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    buf = arr.tobytes(order="C")
    t = torch.frombuffer(buf, dtype=torch.float32).reshape(arr.shape).clone()
    return t


def _torch_to_np_f32(t: torch.Tensor) -> np.ndarray:
    """
    Tensor -> ndarray (float32) without ``Tensor.numpy()`` (same NumPy bridge issue).
    Uses a flat Python list; fine for small coefficient vectors.
    """
    t = t.detach().cpu().float().contiguous()
    shape = tuple(t.shape)
    data = t.reshape(-1).tolist()
    return np.asarray(data, dtype=np.float32).reshape(shape)


def parse_channel_weights(spec: Optional[str], n_ch: int) -> np.ndarray:
    """Comma-separated weights; one value repeats to all channels."""
    if spec is None or str(spec).strip() == "":
        return np.ones(n_ch, dtype=np.float32)
    parts = [float(x) for x in str(spec).replace(" ", "").split(",") if x.strip()]
    if len(parts) == 1:
        return np.full(n_ch, parts[0], dtype=np.float32)
    if len(parts) != n_ch:
        raise ValueError(
            f"--lstm_channel_weights: need 1 or {n_ch} comma-separated values for this field, got {len(parts)}"
        )
    return np.asarray(parts, dtype=np.float32)


def parse_rollout_step_weights(spec: Optional[str], horizon: int) -> np.ndarray:
    """
    Weights for each step in multi-step rollout auxiliary loss (normalized to sum=1 in the loss).
    Special: 'ramp' -> 1,2,...,H; 'late' -> linear 0.5..2.0 emphasizing later steps.
    """
    h = int(horizon)
    if h < 2:
        return np.ones(max(1, h), dtype=np.float32)
    if spec is None or str(spec).strip() == "":
        return np.ones(h, dtype=np.float32)
    s = str(spec).strip().lower()
    if s == "ramp":
        return np.arange(1, h + 1, dtype=np.float32)
    if s == "late":
        return np.linspace(0.5, 2.0, h, dtype=np.float32)
    parts = [float(x) for x in str(spec).replace(" ", "").split(",") if x.strip()]
    if len(parts) == 1:
        return np.full(h, parts[0], dtype=np.float32)
    if len(parts) != h:
        raise ValueError(
            f"--rollout_step_weights: need {h} comma-separated values for horizon={h}, or use ramp/late"
        )
    return np.asarray(parts, dtype=np.float32)


def merge_short_temporal_segments(
    segments: List[Tuple[int, int]], min_len: int
) -> List[Tuple[int, int]]:
    """
    Repeatedly merge the shortest segment with a neighbor until every segment has
    length >= min_len (inclusive end index) or only one segment remains.

    Avoids LSTM training on tiny n_win (e.g. length=12, seq_len=6 → n_win=6) which
    overfits and hurts multi-region rollout.
    """
    m = int(min_len)
    if m <= 1 or not segments:
        return list(segments)
    segs: List[Tuple[int, int]] = [(int(s), int(e)) for s, e in segments]
    while len(segs) > 1:
        lens = [e - s + 1 for s, e in segs]
        if all(L >= m for L in lens):
            return segs
        k = int(np.argmin(lens))
        if lens[k] >= m:
            return segs
        if k + 1 < len(segs):
            segs = segs[:k] + [(segs[k][0], segs[k + 1][1])] + segs[k + 2 :]
        else:
            segs = segs[: k - 1] + [(segs[k - 1][0], segs[k][1])]
    return segs


def _to_float_array(x):
    arr = np.asarray(x)
    if arr.dtype == object:
        # object array: one ndarray per time step
        arr = np.array([np.asarray(v) for v in arr], dtype=np.float32)
    return np.asarray(arr, dtype=np.float32)


def _extract_sdf_timeseries(
    result_item: dict,
    T: int,
    N: int,
    sdf_static: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """
    Build per-frame SDF (T, N). If result NPZ has a time-varying key (sdf, sdf_time, ...), use it;
    else repeat static sdf from geometry (time differences will be zero → warn at runtime if SDF-driven seg).
    """
    st = np.asarray(sdf_static, dtype=np.float32).reshape(-1)
    if st.size != N:
        pad = np.zeros(N, dtype=np.float32)
        m = min(st.size, N)
        pad[:m] = st[:m]
        st = pad
    for key in ("sdf", "sdf_time", "signed_distance", "SDF"):
        if key not in result_item:
            continue
        raw = result_item[key]
        arr = np.asarray(raw)
        if arr.dtype == object:
            try:
                arr = _to_float_array(raw)
            except (ValueError, TypeError):
                continue
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == T and arr.shape[1] == N:
            arr = arr[:, :, 0]
        if arr.ndim == 2 and arr.shape[0] == T and arr.shape[1] == N:
            return arr.astype(np.float32), True
    return np.tile(st.reshape(1, -1), (T, 1)).astype(np.float32), False


def _ensure_time_node_feat(field: np.ndarray) -> np.ndarray:
    """
    Normalize field to shape (T, N, C).
    Supports (T, N, C), object array of (N, C) per step, and simple squeeze.
    """
    field = _to_float_array(field)
    field = np.squeeze(field)
    if field.ndim == 2:
        # (N, C) -> single frame
        field = field[None, ...]
    if field.ndim != 3:
        raise ValueError(f"field must be 3D (T,N,C), got shape={field.shape}")
    return field.astype(np.float32)


class NPZSampleLoader:
    """
    Two-file layout aligned with read_and_visualize_dataset.py:
    - SolverFileData.npz: per-sample dict with xyzd / time / edge (and bc, etc.)
    - ResultFileData_displacement.npz or ResultFileData_stress.npz: displacement or stress

    xyzd[:, :3] are node coordinates; xyzd[:, 3:] are per-point replicated global features.
    """

    def __init__(self, solver_npz_path: str, result_npz_path: str):
        if not os.path.exists(solver_npz_path):
            raise FileNotFoundError(f"File not found: {solver_npz_path}")
        if not os.path.exists(result_npz_path):
            raise FileNotFoundError(f"File not found: {result_npz_path}")
        self.solver_npz_path = solver_npz_path
        self.result_npz_path = result_npz_path
        self.solver_data = np.load(solver_npz_path, allow_pickle=True)
        self.result_data = np.load(result_npz_path, allow_pickle=True)

        self.sample_keys = sorted(
            list(set(self.solver_data.files).intersection(set(self.result_data.files)))
        )
        if not self.sample_keys:
            raise RuntimeError("Solver and result NPZ share no sample keys; check files.")

    def list_samples(self) -> List[str]:
        return self.sample_keys

    def load_sample(self, sample_key: str, target: str) -> Dict[str, np.ndarray]:
        """
        target: 'displacement' | 'stress'; field name from TARGET_CONFIG.
        Displacement: keep only first 3 channels if C > 3 (same as disp_vec = frame[:, :3] in viz).

        Also returns:
          sdf: (N,) column 3 of xyzd (treated as SDF / per-point scalar in dataset convention).
          global_feat: (G,) from xyzd[0, 4:] replicated on all nodes in read_and_visualize_dataset.
        """
        if target not in TARGET_CONFIG:
            raise ValueError(f"Unknown target={target}")
        field_key = TARGET_CONFIG[target]["field_key"]

        if sample_key not in self.solver_data.files:
            raise KeyError(f"{sample_key} missing in solver NPZ: {self.solver_npz_path}")
        if sample_key not in self.result_data.files:
            raise KeyError(f"{sample_key} missing in result NPZ: {self.result_npz_path}")

        solver_item = self.solver_data[sample_key].item()
        result_item = self.result_data[sample_key].item()

        if field_key not in result_item:
            avail = list(result_item.keys())
            raise KeyError(f"Sample {sample_key}: no key '{field_key}'. Available: {avail}")

        # Like read_and_visualize_dataset.main: load float64, then use float32 for training
        xyzd = np.asarray(solver_item["xyzd"], dtype=np.float64)
        time_steps = np.asarray(solver_item["time"])
        time = np.asarray(time_steps, dtype=np.float32).reshape(-1)
        edge = np.asarray(
            solver_item.get("edge", np.zeros((0, 2), dtype=np.int64)),
            dtype=np.int64,
        )

        raw = result_item[field_key]
        arr = np.asarray(raw)
        if arr.dtype == object:
            arr = _to_float_array(raw)
        else:
            arr = arr.astype(np.float64, copy=False)

        field = _ensure_time_node_feat(arr)
        if target == "displacement" and field.shape[-1] > 3:
            field = field[..., :3].copy()

        t_field = field.shape[0]
        t_time = int(time.shape[0])
        if t_field != t_time:
            t_min = min(t_field, t_time)
            field = field[:t_min]
            time = time[:t_min]

        sdf = xyzd[:, 3].astype(np.float32)
        if xyzd.shape[1] > 4:
            global_feat = xyzd[0, 4:].astype(np.float32)
        else:
            global_feat = np.zeros(0, dtype=np.float32)

        t_n, n_nodes = field.shape[0], field.shape[1]
        sdf_seq, sdf_time_varying = _extract_sdf_timeseries(result_item, t_n, n_nodes, sdf)

        return {
            "xyzd": xyzd.astype(np.float32),
            "time": time,
            "edge": edge,
            "field": field.astype(np.float32),
            "sdf": sdf,
            "sdf_seq": sdf_seq.astype(np.float32),
            "sdf_time_varying": sdf_time_varying,
            "global_feat": global_feat,
            "field_key": field_key,
            "component_names": list(TARGET_CONFIG[target]["components"]),
        }


class TemporalSegmenter:
    """Temporal segmentation from field change rate and/or mean(|Δsdf|)/dt."""

    def __init__(
        self,
        threshold_ratio: float = 0.5,
        min_segment_len: int = 5,
        use_physical_time: bool = False,
        use_sdf_seg: bool = False,
    ):
        self.threshold_ratio = threshold_ratio
        self.min_segment_len = min_segment_len
        self.use_physical_time = use_physical_time
        self.use_sdf_seg = use_sdf_seg

    def segment(
        self,
        field_seq: np.ndarray,
        time_seq: Optional[np.ndarray] = None,
        sdf_seq: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, int]]:
        t_total = field_seq.shape[0]
        if t_total < 2:
            return [(0, t_total - 1)]

        diff_norms = []
        use_sdf = (
            self.use_sdf_seg
            and sdf_seq is not None
            and sdf_seq.shape[0] == t_total
            and sdf_seq.shape[1] == field_seq.shape[1]
        )
        for t in range(t_total - 1):
            if use_sdf:
                ds = np.abs(
                    sdf_seq[t + 1].astype(np.float64) - sdf_seq[t].astype(np.float64)
                )
                mean_abs = float(np.mean(ds))
                if time_seq is not None and len(time_seq) > t + 1:
                    dt = max(float(time_seq[t + 1] - time_seq[t]), 1e-12)
                    norm = mean_abs / dt
                else:
                    norm = mean_abs
            else:
                diff = field_seq[t + 1] - field_seq[t]
                if self.use_physical_time and time_seq is not None and len(time_seq) > t + 1:
                    dt = float(time_seq[t + 1] - time_seq[t])
                    if dt > 1e-12:
                        norm = np.linalg.norm(diff) / (np.sqrt(diff.size) * dt)
                    else:
                        norm = np.linalg.norm(diff) / np.sqrt(diff.size)
                else:
                    norm = np.linalg.norm(diff) / np.sqrt(diff.size)
            diff_norms.append(norm)
        diff_norms = np.asarray(diff_norms)

        threshold = diff_norms.mean() + self.threshold_ratio * diff_norms.std()
        break_points = np.where(diff_norms > threshold)[0] + 1

        segments = []
        start = 0
        for bp in break_points:
            if bp - start >= self.min_segment_len:
                segments.append((start, bp - 1))
                start = bp
        if t_total - start >= self.min_segment_len:
            segments.append((start, t_total - 1))
        if not segments:
            segments = [(0, t_total - 1)]
        return segments


class PODReducer:
    """POD reduction for C channels (e.g. displacement C=3, stress C=7)."""

    def __init__(
        self,
        energy_ratio: float = 0.99,
        full_rank: bool = False,
        max_modes: Optional[int] = None,
    ):
        self.energy_ratio = energy_ratio
        self.full_rank = bool(full_rank)
        self.max_modes = max_modes
        self.mean = None  # (N*C, 1)
        self.U = None  # (N*C, r)
        self.n_nodes = None
        self.n_feat = None
        # Singular values / energy on training snapshots (for plots)
        self.singular_values = None  # (n_modes_full,)
        self.cumulative_energy = None  # (n_modes_full,) cumulative sigma^2 energy fraction
        self.n_modes_retained = None  # retained rank r

    def fit(self, snapshots: np.ndarray) -> int:
        # snapshots: (T, N, C)
        t_total, n_nodes, n_feat = snapshots.shape
        self.n_nodes = n_nodes
        self.n_feat = n_feat

        x = snapshots.reshape(t_total, -1).T  # (N*C, T)
        self.mean = x.mean(axis=1, keepdims=True)
        x_centered = x - self.mean

        u, s, _ = np.linalg.svd(x_centered, full_matrices=False)
        total_e = float(np.sum(s ** 2))
        energy = np.cumsum(s ** 2) / (total_e + 1e-12)
        self.singular_values = s.astype(np.float64)
        self.cumulative_energy = energy.astype(np.float64)
        r_cap = int(s.size)
        if self.full_rank:
            r_wish = r_cap
            r = r_cap
        else:
            r_wish = int(np.searchsorted(energy, self.energy_ratio) + 1)
            r = min(r_wish, r_cap)
        r = max(r, 1)
        if self.max_modes is not None and int(self.max_modes) > 0:
            r = min(r, int(self.max_modes))
        self.n_modes_retained = r
        self.U = u[:, :r]
        if r_wish > r_cap:
            print(
                f"[WARN] POD: energy_ratio would keep ~{r_wish} modes, "
                f"but SVD has only {r_cap} non-zero singular values (T={t_total} snapshots cap rank). Using r={r}."
            )
        return r

    def mode_vector(self, k: int) -> np.ndarray:
        """k-th POD basis as flat vector (N*C,), k in [0, U.shape[1])."""
        if self.U is None:
            raise RuntimeError("POD not fitted yet.")
        if k < 0 or k >= self.U.shape[1]:
            raise IndexError(f"Mode index k={k} out of range [0, {self.U.shape[1]})")
        return self.U[:, k].copy()

    def mode_field(self, k: int) -> np.ndarray:
        """k-th mode reshaped to (N, C)."""
        v = self.mode_vector(k)
        return v.reshape(self.n_nodes, self.n_feat)

    def transform(self, snapshots: np.ndarray) -> np.ndarray:
        t_total, _, _ = snapshots.shape
        x = snapshots.reshape(t_total, -1).T
        coeff = self.U.T @ (x - self.mean)  # (r, T)
        return coeff.T  # (T, r)

    def inverse_transform(self, coeff: np.ndarray) -> np.ndarray:
        # coeff: (T, r)
        x_rec = self.U @ coeff.T + self.mean  # (N*C, T)
        return x_rec.T.reshape(coeff.shape[0], self.n_nodes, self.n_feat)


class ChannelwisePODReducer:
    """
    One SVD/POD per scalar channel on (N, T): avoids joint (N·C)-vector modes mixing σ/τ patterns.
    Coefficients are concatenated; U/mean are assembled in the same row order as PODReducer (node-major, C inner).
    """

    def __init__(
        self,
        energy_ratio: float,
        full_rank: bool = False,
        max_modes_per_channel: Optional[int] = None,
    ):
        self.energy_ratio = float(energy_ratio)
        self.full_rank = bool(full_rank)
        self.max_modes_per_channel = max_modes_per_channel
        self.reducers: List[PODReducer] = []
        self.n_nodes: int = 0
        self.n_feat: int = 0
        self.mean = None
        self.U = None
        self.singular_values = None
        self.cumulative_energy = None
        self.n_modes_retained: int = 0

    def fit(self, snapshots: np.ndarray) -> int:
        t_total, n_nodes, n_feat = snapshots.shape
        self.n_nodes = n_nodes
        self.n_feat = n_feat
        self.reducers = []
        total_r = 0
        cap = self.max_modes_per_channel
        ranks: List[int] = []
        for c in range(n_feat):
            sub = snapshots[:, :, c : c + 1]
            pr = PODReducer(
                self.energy_ratio,
                full_rank=self.full_rank,
                max_modes=cap,
            )
            r = pr.fit(sub)
            self.reducers.append(pr)
            total_r += r
            ranks.append(r)
        self._assemble_U_mean()
        self.n_modes_retained = total_r
        self.singular_values = np.concatenate([pr.singular_values for pr in self.reducers])
        self.cumulative_energy = None
        print(
            f"[INFO]   Channelwise stress POD: per-channel ranks {ranks}, total r={total_r} "
            f"(energy={self.energy_ratio}, full_rank={self.full_rank}, cap={cap or 'none'})"
        )
        return total_r

    def _assemble_U_mean(self) -> None:
        n, c_dim = self.n_nodes, self.n_feat
        r_tot = sum(int(r.n_modes_retained) for r in self.reducers)
        u_big = np.zeros((n * c_dim, r_tot), dtype=np.float64)
        mean_big = np.zeros((n * c_dim, 1), dtype=np.float64)
        col = 0
        for c, pr in enumerate(self.reducers):
            r = int(pr.n_modes_retained)
            u_c = np.asarray(pr.U, dtype=np.float64)
            m_c = np.asarray(pr.mean, dtype=np.float64)
            rows = np.arange(n, dtype=np.int64) * c_dim + c
            u_big[rows, col : col + r] = u_c
            mean_big[rows, 0:1] = m_c
            col += r
        self.U = u_big
        self.mean = mean_big

    def transform(self, snapshots: np.ndarray) -> np.ndarray:
        pieces = [
            self.reducers[c].transform(snapshots[:, :, c : c + 1]) for c in range(self.n_feat)
        ]
        return np.concatenate(pieces, axis=1)

    def inverse_transform(self, coeff: np.ndarray) -> np.ndarray:
        t_total = int(coeff.shape[0])
        out = np.zeros((t_total, self.n_nodes, self.n_feat), dtype=np.float64)
        col = 0
        for c, pr in enumerate(self.reducers):
            r = int(pr.n_modes_retained)
            sl = coeff[:, col : col + r]
            out[:, :, c : c + 1] = np.asarray(pr.inverse_transform(sl), dtype=np.float64)
            col += r
        return out.astype(np.float32)

    def mode_vector(self, k: int) -> np.ndarray:
        kk = int(k)
        for c, pr in enumerate(self.reducers):
            r = int(pr.n_modes_retained)
            if kk < r:
                v = pr.mode_vector(kk)
                big = np.zeros(self.n_nodes * self.n_feat, dtype=np.float64)
                rows = np.arange(self.n_nodes, dtype=np.int64) * self.n_feat + c
                big[rows] = v
                return big
            kk -= r
        raise IndexError(f"Mode index {k} out of range for channelwise POD")

    def mode_field(self, k: int) -> np.ndarray:
        return self.mode_vector(k).reshape(self.n_nodes, self.n_feat)


class LSTMDynamics(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        out_d = output_dim if output_dim is not None else input_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, out_d)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _stress_scale_field_in(fld: np.ndarray, stress_std: Optional[np.ndarray]) -> np.ndarray:
    """Divide physical stress field by per-component std before POD transform."""
    x = fld.astype(np.float32)
    if stress_std is None:
        return x
    s = stress_std.reshape(1, 1, -1).astype(np.float32)
    return x / s


def _stress_unscale_field_out(fld: np.ndarray, stress_std: Optional[np.ndarray]) -> np.ndarray:
    """Map POD inverse output back to physical stress (multiply by per-component std)."""
    if stress_std is None:
        return fld.astype(np.float32)
    return (fld * stress_std.reshape(1, -1).astype(np.float32)).astype(np.float32)


@dataclass
class SegmentModel:
    reducer: Any  # PODReducer or ChannelwisePODReducer
    lstm: Optional[LSTMDynamics]
    length: int
    rank: int
    coeff_mean: Optional[np.ndarray] = None  # (r,) physical POD space, for LSTM I/O denorm
    coeff_std: Optional[np.ndarray] = None
    seq_eff: int = 1  # history length before left-padding to seq_len (matches training)
    time_tmin: Optional[float] = None  # segment physical time range for normalizing time feature
    time_tmax: Optional[float] = None
    predict_delta: bool = False  # LSTM predicts Δcoeff_n; rollout adds to last normalized coeff
    stress_channel_std: Optional[np.ndarray] = None  # (C,) if set, POD was on field/std; inverse → ×std


class SegmentedROM:
    def __init__(
        self,
        energy_ratio: float = 0.99,
        seq_len: int = 5,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        threshold_ratio: float = 0.5,
        min_segment_len: int = 5,
        normalize_coeffs: bool = True,
        use_physical_time_seg: bool = False,
        use_time_feature: bool = False,
        use_sdf_segmentation: bool = False,
        pod_full_rank: bool = False,
    ):
        self.energy_ratio = energy_ratio
        self.pod_full_rank = bool(pod_full_rank)
        self.seq_len = seq_len
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.normalize_coeffs = normalize_coeffs
        self.use_time_feature = use_time_feature
        self.segmenter = TemporalSegmenter(
            threshold_ratio,
            min_segment_len,
            use_physical_time=use_physical_time_seg,
            use_sdf_seg=use_sdf_segmentation,
        )

        self.segments: List[Tuple[int, int]] = []
        self.models: List[SegmentModel] = []
        self.global_feat: Optional[np.ndarray] = None
        self.global_feat_dim: int = 0
        self.time_series_train: Optional[np.ndarray] = None
        self.dt_forecast: float = 1.0

    @staticmethod
    def _choose_seq_eff(n_frames: int, seq_cap: int) -> int:
        """
        Shorter effective history when the segment has few frames so we get enough supervision windows.
        Example: L=5, seq_cap=4 -> naive n_win=1; reduce to seq_eff=2 -> n_win=3.
        """
        if n_frames <= 2:
            return 1
        seq_eff = min(seq_cap, n_frames - 1)
        n_win = n_frames - seq_eff
        if n_win < 3 and seq_eff > 1:
            seq_eff = max(1, n_frames - 3)
        return int(min(max(1, seq_eff), n_frames - 1))

    def _build_lstm_features_matrix(
        self, coeff_n: np.ndarray, time_sub: np.ndarray
    ) -> np.ndarray:
        """Per time row: [coeff_n | time_norm (optional) | global_feat]."""
        L, r = coeff_n.shape
        if self.use_time_feature and time_sub.size == L:
            tmin = float(time_sub.min())
            tmax = float(time_sub.max())
            if tmax > tmin + 1e-12:
                time_col = ((time_sub.astype(np.float64) - tmin) / (tmax - tmin)).reshape(-1, 1)
            else:
                time_col = np.zeros((L, 1), dtype=np.float64)
            time_col = time_col.astype(np.float32)
        else:
            time_col = np.zeros((L, 0), dtype=np.float32)
        if self.global_feat_dim > 0 and self.global_feat is not None:
            gf = np.repeat(self.global_feat.reshape(1, -1).astype(np.float32), L, axis=0)
        else:
            gf = np.zeros((L, 0), dtype=np.float32)
        return np.concatenate([coeff_n.astype(np.float32), time_col, gf], axis=1)

    def _build_lstm_dataset_padded(
        self,
        coeff_n: np.ndarray,
        time_sub: np.ndarray,
        seq_eff: int,
        seq_pad: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        features = self._build_lstm_features_matrix(coeff_n, time_sub)
        _, input_dim = features.shape
        r = coeff_n.shape[1]
        x_list, y_list = [], []
        for i in range(seq_eff, coeff_n.shape[0]):
            hist = features[i - seq_eff : i]
            if hist.shape[0] < seq_pad:
                pad = seq_pad - hist.shape[0]
                hist = np.vstack([np.tile(hist[0:1], (pad, 1)), hist])
            x_list.append(hist)
            y_list.append(coeff_n[i])
        if not x_list:
            return (
                np.empty((0, seq_pad, input_dim), dtype=np.float32),
                np.empty((0, r), dtype=np.float32),
            )
        return np.asarray(x_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)

    def _rollout_auxiliary_loss(
        self,
        model: LSTMDynamics,
        coeff_n: np.ndarray,
        time_sub: np.ndarray,
        seq_eff: int,
        seq_pad: int,
        tmin_s: float,
        tmax_s: float,
        device: torch.device,
        horizon: int,
        teacher_forcing_ratio: float,
        predict_delta: bool,
        rng: np.random.Generator,
        step_weights: np.ndarray,
        coeff_smooth_l1: bool,
    ) -> torch.Tensor:
        """Multi-step loss along one random sub-trajectory; teacher forcing mixes true vs predicted history."""
        L, _r = coeff_n.shape
        if horizon < 2 or L < seq_eff + horizon:
            return torch.zeros((), device=device, dtype=torch.float32)
        w_np = np.asarray(step_weights, dtype=np.float64).reshape(-1)
        if w_np.size != horizon:
            raise ValueError(f"rollout step_weights length {w_np.size} != horizon {horizon}")
        w_t = torch.as_tensor(w_np / max(float(w_np.sum()), 1e-8), device=device, dtype=torch.float32)
        t0 = int(rng.integers(seq_eff, L - horizon + 1))
        coeff_np = coeff_n.astype(np.float32)
        hist = [
            torch.as_tensor(coeff_np[t0 - seq_eff + j], device=device, dtype=torch.float32)
            for j in range(seq_eff)
        ]
        span_t = max(float(tmax_s - tmin_s), 1e-12)
        total = torch.zeros((), device=device, dtype=torch.float32)
        for h in range(horizon):
            curr = t0 + h
            c_stack = torch.stack(hist[-seq_eff:], dim=0)
            time_idx = np.arange(curr - seq_eff, curr, dtype=np.int64)
            times = time_sub[time_idx].astype(np.float64)
            if self.use_time_feature:
                tn = torch.as_tensor(
                    ((times - tmin_s) / span_t).reshape(-1, 1), device=device, dtype=torch.float32
                )
            else:
                tn = torch.zeros(seq_eff, 0, device=device, dtype=torch.float32)
            if self.global_feat_dim > 0 and self.global_feat is not None:
                gf = torch.as_tensor(self.global_feat, device=device, dtype=torch.float32).reshape(1, -1)
                gf = gf.expand(seq_eff, -1)
            else:
                gf = torch.zeros(seq_eff, 0, device=device, dtype=torch.float32)
            feat = torch.cat([c_stack, tn, gf], dim=1)
            if feat.shape[0] < seq_pad:
                pad_n = seq_pad - feat.shape[0]
                feat = torch.cat([feat[0:1].expand(pad_n, -1), feat], dim=0)
            elif feat.shape[0] > seq_pad:
                feat = feat[-seq_pad:]
            pred = model(feat.unsqueeze(0)).squeeze(0)
            if predict_delta:
                tgt_np = coeff_np[curr] - coeff_np[curr - 1]
            else:
                tgt_np = coeff_np[curr]
            tgt = torch.as_tensor(tgt_np, device=device, dtype=torch.float32)
            if coeff_smooth_l1:
                step_l = F.smooth_l1_loss(pred, tgt, beta=1.0, reduction="mean")
            else:
                step_l = torch.mean((pred - tgt) ** 2)
            total = total + w_t[h] * step_l
            if float(rng.random()) < float(teacher_forcing_ratio):
                nxt = torch.as_tensor(coeff_np[curr], device=device, dtype=torch.float32)
            else:
                if predict_delta:
                    nxt = hist[-1] + pred.detach()
                else:
                    nxt = pred.detach()
            hist.append(nxt)
            hist = hist[-seq_eff:]
        return total

    def fit(
        self,
        field_seq: np.ndarray,
        time_series: np.ndarray,
        global_feat: Optional[np.ndarray],
        device: torch.device,
        sdf_seq: Optional[np.ndarray] = None,
        epochs: int = 120,
        lr: float = 1e-3,
        batch_size: int = 32,
        weight_decay: float = 0.0,
        lr_scheduler: str = "none",
        grad_clip: float = 0.0,
        cosine_eta_min: Optional[float] = None,
        lstm_field_loss_ratio: float = 0.0,
        lstm_channel_weights: Optional[np.ndarray] = None,
        lstm_predict_delta: bool = False,
        train_rollout_horizon: int = 1,
        teacher_forcing_ratio: float = 1.0,
        rollout_loss_weight: float = 0.0,
        rollout_sample_prob: float = 0.5,
        rollout_aux_seed: int = 0,
        tf_ratio_start: Optional[float] = None,
        rollout_step_weights: Optional[str] = None,
        lstm_coeff_smoothl1: bool = False,
        lstm_ema_decay: float = 0.0,
        merge_short_segments_len: int = 0,
        stress_per_channel_std: bool = False,
        stress_pod_channelwise: bool = False,
        stress_pod_channelwise_cap: int = 0,
    ) -> None:
        time_series = np.asarray(time_series, dtype=np.float64).reshape(-1)
        self.time_series_train = time_series.astype(np.float64)
        if global_feat is not None and np.size(global_feat) > 0:
            self.global_feat = np.asarray(global_feat, dtype=np.float32).reshape(-1)
            self.global_feat_dim = int(self.global_feat.shape[0])
        else:
            self.global_feat = None
            self.global_feat_dim = 0

        diffs = np.diff(time_series)
        self.dt_forecast = float(np.mean(diffs)) if diffs.size else 1.0

        if self.segmenter.use_sdf_seg:
            if (
                sdf_seq is None
                or sdf_seq.shape[0] != field_seq.shape[0]
                or sdf_seq.shape[1] != field_seq.shape[1]
            ):
                print(
                    "[WARN] SDF segmentation: invalid sdf_seq shape vs field_seq; "
                    "using displacement-based segmentation instead."
                )
                self.segmenter.use_sdf_seg = False

        if self.segmenter.use_sdf_seg:
            ts_for_seg = time_series
            sdf_pass = sdf_seq
        else:
            ts_for_seg = time_series if self.segmenter.use_physical_time else None
            sdf_pass = None

        self.segments = self.segmenter.segment(field_seq, ts_for_seg, sdf_pass)
        seg_metric = "mean(|Δsdf|)/dt" if self.segmenter.use_sdf_seg else "displacement rate"
        print(f"[INFO] Segments ({seg_metric}, raw): {self.segments}")
        msl = int(merge_short_segments_len)
        if msl > 1:
            merged = merge_short_temporal_segments(self.segments, msl)
            if merged != self.segments:
                print(
                    f"[INFO] Temporal segment merge: min_len={msl} → {len(self.segments)} segment(s) "
                    f"reduced to {len(merged)}: {merged}"
                )
            self.segments = merged
        self.models = []
        print(f"[INFO] Segments (final): {self.segments}")

        for idx, (st, ed) in enumerate(self.segments):
            seg = field_seq[st : ed + 1]
            seg_len = seg.shape[0]
            time_sub = time_series[st : ed + 1].astype(np.float64)
            rng_roll = np.random.default_rng((int(rollout_aux_seed) + idx * 100003) % (2**31))
            print(f"\n[INFO] Segment {idx + 1}/{len(self.segments)}: frames {st}–{ed}, length={seg_len}")

            n_ch = int(seg.shape[2])
            n_stress = len(TARGET_CONFIG["stress"]["components"])
            stress_ch_std: Optional[np.ndarray] = None
            seg_pod = seg
            if stress_per_channel_std and n_ch == n_stress:
                stress_ch_std = seg.reshape(-1, n_ch).std(axis=0).astype(np.float64)
                stress_ch_std = np.maximum(stress_ch_std, 1e-12)
                seg_pod = (seg.astype(np.float64) / stress_ch_std.reshape(1, 1, -1)).astype(np.float32)
                print(f"[INFO]   Stress POD input: per-channel std (train segment) = {stress_ch_std}")

            use_ch_pod = bool(stress_pod_channelwise) and n_ch == n_stress
            cap_ch = int(stress_pod_channelwise_cap)
            per_ch_cap = cap_ch if cap_ch > 0 else None
            if use_ch_pod:
                reducer = ChannelwisePODReducer(
                    self.energy_ratio,
                    full_rank=self.pod_full_rank,
                    max_modes_per_channel=per_ch_cap,
                )
            else:
                reducer = PODReducer(self.energy_ratio, full_rank=self.pod_full_rank)
            rank = reducer.fit(seg_pod)
            coeff = reducer.transform(seg_pod)
            print(f"[INFO]   POD rank r={rank}")

            if self.normalize_coeffs:
                coeff_mean = coeff.mean(axis=0).astype(np.float64)
                coeff_std = np.maximum(coeff.std(axis=0), 1e-6).astype(np.float64)
                coeff_n = ((coeff - coeff_mean) / coeff_std).astype(np.float32)
            else:
                coeff_mean = np.zeros(rank, dtype=np.float64)
                coeff_std = np.ones(rank, dtype=np.float64)
                coeff_n = coeff.astype(np.float32)

            tmin_s = float(time_sub.min())
            tmax_s = float(time_sub.max())

            seq_eff = self._choose_seq_eff(seg_len, self.seq_len)
            x_train, y_train = self._build_lstm_dataset_padded(
                coeff_n, time_sub.astype(np.float64), seq_eff, self.seq_len
            )
            if lstm_predict_delta:
                Lc = coeff_n.shape[0]
                y_train = np.stack(
                    [coeff_n[i] - coeff_n[i - 1] for i in range(seq_eff, Lc)], axis=0
                ).astype(np.float32)
            n_win = x_train.shape[0]
            hrz = int(train_rollout_horizon)
            rlw = float(rollout_loss_weight)
            tfr = float(np.clip(teacher_forcing_ratio, 0.0, 1.0))
            rsp = float(np.clip(rollout_sample_prob, 0.0, 1.0))
            rstep_w = parse_rollout_step_weights(rollout_step_weights, hrz)
            tf_sched = tf_ratio_start is not None
            print(
                f"[INFO]   LSTM windows: n_win={n_win}, seq_eff={seq_eff} (padded to seq_len={self.seq_len}), "
                f"coeff_norm={'on' if self.normalize_coeffs else 'off'}, "
                f"time_feature={'on' if self.use_time_feature else 'off'}, "
                f"global_dim={self.global_feat_dim}, "
                f"predict_delta={lstm_predict_delta}, "
                f"rollout_H={hrz}, tf_end={tfr}, tf_schedule={'linear '+str(tf_ratio_start)+'→'+str(tfr) if tf_sched else 'const'}, "
                f"rollout_w={rlw}, rollout_step_w={rstep_w.tolist() if hrz >= 2 else 'n/a'}, "
                f"coeff_loss={'SmoothL1' if lstm_coeff_smoothl1 else 'MSE'}, "
                f"ema_decay={float(lstm_ema_decay)}"
            )
            if 0 < n_win < 12:
                print(
                    f"[WARN]   n_win={n_win} is very small for an LSTM (segment length={seg_len}); "
                    f"large --hidden/--layers or many epochs → memorize training windows, rollout explodes. "
                    f"Try: larger --min_segment_len, higher --threshold_ratio (fewer segments), "
                    f"--lr_scheduler none, smaller --hidden/--seq_len, or fewer --epochs."
                )
            if n_win == 0:
                print(f"[WARN]   Segment too short for LSTM; skipping LSTM")
                self.models.append(
                    SegmentModel(
                        reducer=reducer,
                        lstm=None,
                        length=seg_len,
                        rank=rank,
                        coeff_mean=None,
                        coeff_std=None,
                        seq_eff=1,
                        time_tmin=None,
                        time_tmax=None,
                        predict_delta=bool(lstm_predict_delta),
                        stress_channel_std=(
                            stress_ch_std.astype(np.float32) if stress_ch_std is not None else None
                        ),
                    )
                )
                continue

            ds = TensorDataset(_np_to_torch_f32(x_train), _np_to_torch_f32(y_train))
            dl = DataLoader(ds, batch_size=min(batch_size, max(1, n_win)), shuffle=True)

            input_dim = x_train.shape[2]
            model = LSTMDynamics(
                input_dim, self.lstm_hidden, self.lstm_layers, output_dim=rank
            ).to(device)
            wd = float(weight_decay)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd if wd > 0 else 0.0)
            criterion = nn.SmoothL1Loss() if lstm_coeff_smoothl1 else nn.MSELoss()
            a_field = float(np.clip(lstm_field_loss_ratio, 0.0, 1.0))
            n_feat = int(reducer.n_feat)
            n_nodes = int(reducer.n_nodes)
            ch_w = lstm_channel_weights
            if ch_w is None:
                ch_w = np.ones(n_feat, dtype=np.float32)
            ch_w = np.asarray(ch_w, dtype=np.float32).reshape(-1)
            if ch_w.shape[0] != n_feat:
                raise ValueError(f"lstm_channel_weights length {ch_w.shape[0]} != n_feat={n_feat}")
            if a_field > 0:
                U_t = _np_to_torch_f32(np.ascontiguousarray(reducer.U.astype(np.float32))).to(device)
                mean_t = _np_to_torch_f32(
                    np.ascontiguousarray(reducer.mean.astype(np.float32).reshape(-1))
                ).to(device)
                c_mean_t = _np_to_torch_f32(np.ascontiguousarray(coeff_mean.astype(np.float32))).to(
                    device
                ).unsqueeze(0)
                c_std_t = _np_to_torch_f32(np.ascontiguousarray(coeff_std.astype(np.float32))).to(
                    device
                ).unsqueeze(0)
                w_t = _np_to_torch_f32(np.ascontiguousarray(ch_w)).to(device).view(1, 1, -1)
                w_sum = float(np.sum(ch_w))
                print(
                    f"[INFO]   LSTM loss blend: coeff * {1.0 - a_field:.3f} + weighted field * {a_field:.3f}, "
                    f"channel_w={ch_w.tolist()}"
                )
            else:
                U_t = mean_t = c_mean_t = c_std_t = w_t = None
                w_sum = 1.0

            sched: Any = None
            if lr_scheduler == "cosine":
                eta_min = cosine_eta_min
                if eta_min is None:
                    eta_min = max(1e-8, float(lr) * 0.05)
                else:
                    eta_min = float(eta_min)
                sched = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, epochs), eta_min=eta_min
                )
            elif lr_scheduler != "none":
                raise ValueError(f"Unknown lr_scheduler={lr_scheduler!r} (use none or cosine)")

            ema_decay = float(np.clip(lstm_ema_decay, 0.0, 0.999999))
            ema_shadow: Dict[str, torch.Tensor] = {}

            model.train()
            for ep in range(epochs):
                if tf_sched:
                    prog = float(ep) / float(max(epochs - 1, 1))
                    tfr_ep = float(tf_ratio_start) + (tfr - float(tf_ratio_start)) * prog
                else:
                    tfr_ep = tfr
                tfr_ep = float(np.clip(tfr_ep, 0.0, 1.0))
                total_loss = 0.0
                for xb, yb in dl:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss_c = criterion(pred, yb)
                    rnk = int(rank)
                    if a_field > 0 and U_t is not None:
                        last_n = xb[:, -1, :rnk]
                        if lstm_predict_delta:
                            pred_nn = last_n + pred
                            true_nn = last_n + yb
                        else:
                            pred_nn = pred
                            true_nn = yb
                        c_phys = pred_nn * c_std_t + c_mean_t
                        y_phys = true_nn * c_std_t + c_mean_t
                        bsz = int(pred.shape[0])
                        rec_p = torch.matmul(c_phys, U_t.T) + mean_t
                        rec_t = torch.matmul(y_phys, U_t.T) + mean_t
                        rec_p = rec_p.view(bsz, n_nodes, n_feat)
                        rec_t = rec_t.view(bsz, n_nodes, n_feat)
                        diff2 = (rec_p - rec_t) ** 2
                        loss_f = (diff2 * w_t).sum() / (bsz * n_nodes * max(w_sum, 1e-8))
                        loss = (1.0 - a_field) * loss_c + a_field * loss_f
                    else:
                        loss = loss_c
                    if hrz >= 2 and rlw > 0 and rng_roll.random() < rsp:
                        rloss = self._rollout_auxiliary_loss(
                            model,
                            coeff_n,
                            time_sub.astype(np.float64),
                            seq_eff,
                            self.seq_len,
                            tmin_s,
                            tmax_s,
                            device,
                            hrz,
                            tfr_ep,
                            lstm_predict_delta,
                            rng_roll,
                            rstep_w,
                            lstm_coeff_smoothl1,
                        )
                        loss = loss + rlw * rloss
                    loss.backward()
                    gc = float(grad_clip)
                    if gc > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), gc)
                    optimizer.step()
                    if ema_decay > 0:
                        with torch.no_grad():
                            for name, p in model.named_parameters():
                                if name not in ema_shadow:
                                    ema_shadow[name] = p.detach().clone()
                                else:
                                    ema_shadow[name].mul_(ema_decay).add_(p.detach(), alpha=1.0 - ema_decay)
                    total_loss += loss.item() * xb.size(0)
                if sched is not None:
                    sched.step()
                if (ep + 1) % max(1, epochs // 5) == 0:
                    mse_norm = total_loss / len(ds)
                    lr_now = optimizer.param_groups[0]["lr"]
                    extra = f", lr={lr_now:.2e}" if sched is not None else ""
                    print(f"[INFO]   epoch {ep + 1:4d}/{epochs}, mse_norm={mse_norm:.6e}{extra}")

            if ema_decay > 0 and ema_shadow:
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if name in ema_shadow:
                            p.copy_(ema_shadow[name])
                print(f"[INFO]   LSTM inference weights = EMA shadow (decay={ema_decay})")

            self.models.append(
                SegmentModel(
                    reducer=reducer,
                    lstm=model,
                    length=seg_len,
                    rank=rank,
                    coeff_mean=coeff_mean.astype(np.float32),
                    coeff_std=coeff_std.astype(np.float32),
                    seq_eff=seq_eff,
                    time_tmin=tmin_s,
                    time_tmax=tmax_s,
                    predict_delta=bool(lstm_predict_delta),
                    stress_channel_std=(
                        stress_ch_std.astype(np.float32) if stress_ch_std is not None else None
                    ),
                )
            )

    def predict(
        self,
        initial_field: np.ndarray,
        n_steps: int,
        device: torch.device,
        t_start: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Model not trained; call fit() first.")

        dt_use = float(dt) if dt is not None else float(self.dt_forecast)
        if t_start is None:
            t_start = (
                float(self.time_series_train[0])
                if self.time_series_train is not None and self.time_series_train.size
                else 0.0
            )
        t_start = float(t_start)

        pred_fields = [initial_field.astype(np.float32)]
        seg_lengths = [m.length for m in self.models]
        seg_cum = np.cumsum(seg_lengths)
        cur_seg = 0

        init_snap = _stress_scale_field_in(
            initial_field[None, ...], self.models[0].stress_channel_std
        )
        coeff0 = self.models[0].reducer.transform(init_snap)[0]
        coeff_hist = [coeff0]
        time_hist = [t_start]

        for step in range(n_steps):
            if (step + 1) >= seg_cum[cur_seg] and (cur_seg + 1) < len(self.models):
                cur_seg += 1
                last_field = _stress_scale_field_in(
                    pred_fields[-1][None, ...], self.models[cur_seg].stress_channel_std
                )
                coeff_new = self.models[cur_seg].reducer.transform(last_field)[0]
                coeff_hist = [coeff_new]
                time_hist = [float(time_hist[-1])]

            seg_model = self.models[cur_seg]
            if seg_model.lstm is None:
                next_coeff = coeff_hist[-1]
            else:
                mean = seg_model.coeff_mean
                std = seg_model.coeff_std
                eff = seg_model.seq_eff
                tail_c = coeff_hist[-eff:] if len(coeff_hist) >= eff else coeff_hist[:]
                hist_phys = np.stack(tail_c, axis=0).astype(np.float32)
                tail_t = time_hist[-eff:] if len(time_hist) >= eff else time_hist[:]
                time_tail = np.asarray(tail_t, dtype=np.float64)
                while hist_phys.shape[0] < eff:
                    hist_phys = np.vstack([hist_phys[0:1], hist_phys])
                    time_tail = np.concatenate([[time_tail[0]], time_tail])
                hist_n = (hist_phys - mean) / std

                if self.use_time_feature and seg_model.time_tmax is not None and seg_model.time_tmin is not None:
                    span = max(seg_model.time_tmax - seg_model.time_tmin, 1e-12)
                    time_norm = ((time_tail - seg_model.time_tmin) / span).reshape(-1, 1).astype(np.float32)
                else:
                    time_norm = np.zeros((eff, 0), dtype=np.float32)

                if self.global_feat_dim > 0 and self.global_feat is not None:
                    gf_rep = np.repeat(self.global_feat.reshape(1, -1).astype(np.float32), eff, axis=0)
                else:
                    gf_rep = np.zeros((eff, 0), dtype=np.float32)

                features = np.concatenate([hist_n, time_norm, gf_rep], axis=1)
                if features.shape[0] < self.seq_len:
                    pad = self.seq_len - features.shape[0]
                    features = np.vstack([np.tile(features[0:1], (pad, 1)), features])
                elif features.shape[0] > self.seq_len:
                    features = features[-self.seq_len :]

                xt = _np_to_torch_f32(features[None, ...]).to(device)
                seg_model.lstm.eval()
                with torch.no_grad():
                    out = seg_model.lstm(xt)
                    next_raw = _torch_to_np_f32(out[0])
                    if seg_model.predict_delta:
                        last_n = hist_n[-1].astype(np.float32)
                        next_n = last_n + next_raw
                    else:
                        next_n = next_raw
                    next_coeff = next_n * std + mean

            coeff_hist.append(next_coeff)
            time_hist.append(float(time_hist[-1] + dt_use))
            next_scaled = seg_model.reducer.inverse_transform(next_coeff[None, :])[0]
            next_field = _stress_unscale_field_out(next_scaled, seg_model.stress_channel_std)
            pred_fields.append(next_field.astype(np.float32))

        return np.asarray(pred_fields, dtype=np.float32)


def spatial_partition_equal_bins(
    coords_xyz: np.ndarray, axis: int, n_regions: int, seed: int = 0
) -> List[np.ndarray]:
    """
    Disjoint node groups: sort nodes by coords[:, axis], split into n_regions slices of
    nearly equal count (bands perpendicular to axis). For crush along y use axis=1 so
    top vs bottom body get separate POD/LSTM stacks (no interface coupling in this version).
    """
    if n_regions < 1:
        raise ValueError("n_regions must be >= 1")
    coords_xyz = np.asarray(coords_xyz, dtype=np.float64)
    n = coords_xyz.shape[0]
    if n_regions == 1:
        return [np.arange(n, dtype=np.int64)]
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0 (x), 1 (y), or 2 (z)")
    order = np.argsort(coords_xyz[:, axis], kind="mergesort")
    cuts = np.array_split(order, n_regions)
    return [np.asarray(c, dtype=np.int64) for c in cuts]


def _normalize_partition_order(order: str) -> str:
    """Permutation of x,y,z for nested 3D splits; default yxz = y first (typical crush stratification)."""
    s = order.lower().replace("-", "").replace(",", "").strip()
    if len(s) != 3 or set(s) != {"x", "y", "z"}:
        return "yxz"
    return s


def spatial_partition_3d_nested(
    coords_xyz: np.ndarray,
    gx: int,
    gy: int,
    gz: int,
    order: str = "yxz",
) -> List[np.ndarray]:
    """
    True 3D partitioning: nested equal-count cuts along x,y,z. ``order`` gives the sequence of
    axes (e.g. ``yxz`` splits along y into ``gy`` bands first, then each along x into ``gx``,
    then each along z into ``gz``), producing up to ``gx*gy*gz`` disjoint node groups (~equal size).

    Uses full 3D coordinates from ``xyzd[:, :3]``; diagnostic scatter plots may still project to XY.
    """
    coords_xyz = np.asarray(coords_xyz, dtype=np.float64)
    n = coords_xyz.shape[0]
    if n == 0:
        return []
    gx, gy, gz = max(1, int(gx)), max(1, int(gy)), max(1, int(gz))
    order = _normalize_partition_order(order)
    axis_bins = {"x": (0, gx), "y": (1, gy), "z": (2, gz)}

    def split_group(indices: np.ndarray, axis: int, n_bins: int) -> List[np.ndarray]:
        if indices.size == 0:
            return []
        if n_bins <= 1:
            return [indices.astype(np.int64, copy=False)]
        sub = coords_xyz[indices, axis]
        o = np.argsort(sub, kind="mergesort")
        parts = np.array_split(indices[o], n_bins)
        out = [p.astype(np.int64, copy=False) for p in parts if p.size > 0]
        return out

    groups: List[np.ndarray] = [np.arange(n, dtype=np.int64)]
    for char in order:
        axis, nb = axis_bins[char]
        new_groups: List[np.ndarray] = []
        for g in groups:
            new_groups.extend(split_group(g, axis, nb))
        groups = new_groups

    return [np.sort(g.astype(np.int64)) for g in groups if g.size > 0]


def parse_spatial_grid(spec: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in spec.replace(" ", "").split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("--spatial_grid must be gx,gy,gz with three integers, e.g. 2,2,2")
    return max(1, int(parts[0])), max(1, int(parts[1])), max(1, int(parts[2]))


class SpatioTemporalROM:
    """
    Space–time partitioned ROM: each spatial region has its own SegmentedROM (temporal
    segments + POD + LSTM). Regions may come from 1D bands or 3D nested ``gx,gy,gz`` boxes
    (see ``spatial_partition_3d_nested``). Fields are stitched by node index; interface
    physics between regions is not enforced.
    """

    def __init__(
        self,
        region_indices: List[np.ndarray],
        energy_ratio: float,
        seq_len: int,
        lstm_hidden: int,
        lstm_layers: int,
        threshold_ratio: float,
        min_segment_len: int,
        normalize_coeffs: bool,
        use_physical_time_seg: bool = False,
        use_time_feature: bool = False,
        use_sdf_segmentation: bool = False,
        pod_full_rank: bool = False,
    ):
        self.region_indices = region_indices
        self.n_regions = len(region_indices)
        self.roms: List[SegmentedROM] = [
            SegmentedROM(
                energy_ratio=energy_ratio,
                seq_len=seq_len,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                threshold_ratio=threshold_ratio,
                min_segment_len=min_segment_len,
                normalize_coeffs=normalize_coeffs,
                use_physical_time_seg=use_physical_time_seg,
                use_time_feature=use_time_feature,
                use_sdf_segmentation=use_sdf_segmentation,
                pod_full_rank=pod_full_rank,
            )
            for _ in range(self.n_regions)
        ]

    def fit(
        self,
        field_seq: np.ndarray,
        time_series: np.ndarray,
        global_feat: Optional[np.ndarray],
        sdf_seq: Optional[np.ndarray],
        device: torch.device,
        **kwargs: Any,
    ) -> None:
        base_roll = int(kwargs.pop("rollout_aux_seed", 0))
        for r, idx in enumerate(self.region_indices):
            print(f"\n[INFO] ========== Spatial region {r + 1}/{self.n_regions} ({idx.size} nodes) ==========")
            sub = field_seq[:, idx, :].copy()
            sdf_sub = sdf_seq[:, idx].copy() if sdf_seq is not None else None
            kw = {**kwargs, "rollout_aux_seed": base_roll + r * 1009}
            self.roms[r].fit(
                sub,
                time_series,
                global_feat,
                device=device,
                sdf_seq=sdf_sub,
                **kw,
            )

    def predict(
        self,
        initial_field: np.ndarray,
        n_steps: int,
        device: torch.device,
        t_start: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        n_nodes, n_ch = initial_field.shape
        out = np.zeros((n_steps + 1, n_nodes, n_ch), dtype=np.float32)
        out[0] = initial_field.astype(np.float32)
        region_preds: List[np.ndarray] = []
        for r, idx in enumerate(self.region_indices):
            sub_ic = initial_field[idx, :].astype(np.float32)
            pred_r = self.roms[r].predict(
                sub_ic, n_steps, device=device, t_start=t_start, dt=dt
            )
            region_preds.append(pred_r)
        for t in range(n_steps + 1):
            for r, idx in enumerate(self.region_indices):
                out[t, idx, :] = region_preds[r][t]
        return out


def _subsample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def _safe_plot_token(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))


def plot_pod_segment_analysis(
    reducer: Any,
    segment_idx: int,
    coords_xyz: np.ndarray,
    out_prefix: str,
    n_modes_spatial: int = 6,
    scatter_max_points: int = 40000,
    subsample_seed: int = 0,
    channel_names: Optional[List[str]] = None,
) -> None:
    """
    Visualize POD for one temporal segment:
    (1) singular values and cumulative energy
    (2) first spatial modes (scatter on XY, |phi| colormap)
    """
    if isinstance(reducer, ChannelwisePODReducer):
        names = channel_names or [f"c{i}" for i in range(len(reducer.reducers))]
        for ci, pr in enumerate(reducer.reducers):
            tag = _safe_plot_token(names[ci] if ci < len(names) else f"c{ci}")
            plot_pod_segment_analysis(
                pr,
                segment_idx,
                coords_xyz,
                f"{out_prefix}_ch{tag}",
                n_modes_spatial=n_modes_spatial,
                scatter_max_points=scatter_max_points,
                subsample_seed=subsample_seed,
            )
        return

    if reducer.U is None or reducer.singular_values is None:
        return

    s = reducer.singular_values
    cum_e = reducer.cumulative_energy
    r_keep = reducer.n_modes_retained

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    modes = np.arange(1, len(s) + 1)
    axes[0].semilogy(modes, s, "b.-", markersize=4)
    axes[0].axvline(r_keep, color="r", linestyle="--", linewidth=1, label=f"retained r={r_keep}")
    axes[0].set_xlabel("Mode index")
    axes[0].set_ylabel(r"Singular value $\sigma_k$")
    axes[0].set_title(f"Segment {segment_idx}: singular values")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(modes, cum_e, "g.-", markersize=4)
    axes[1].axhline(reducer.energy_ratio, color="k", linestyle=":", linewidth=1, label=f"target={reducer.energy_ratio}")
    axes[1].axvline(r_keep, color="r", linestyle="--", linewidth=1, label=f"retained r={r_keep}")
    axes[1].set_xlabel("Mode index")
    axes[1].set_ylabel("Cumulative energy fraction")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title(f"Segment {segment_idx}: POD energy spectrum")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_segment{segment_idx}_spectrum.png", dpi=150)
    plt.close(fig)

    xyz = np.asarray(coords_xyz, dtype=np.float64)
    if xyz.shape[0] != reducer.n_nodes:
        # Skip spatial plot if node count mismatch
        return

    n_show = min(n_modes_spatial, reducer.U.shape[1])
    idx = _subsample_indices(reducer.n_nodes, scatter_max_points, subsample_seed)
    xplot = xyz[idx, 0]
    yplot = xyz[idx, 1]

    ncols = min(3, n_show)
    nrows = int(np.ceil(n_show / ncols))
    fig2, axs = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    for k in range(n_show):
        phi = reducer.mode_field(k)[idx]
        amp = np.linalg.norm(phi, axis=1)
        ax = axs[k // ncols][k % ncols]
        sc = ax.scatter(xplot, yplot, c=amp, s=2, cmap="coolwarm", rasterized=True)
        plt.colorbar(sc, ax=ax, label=r"$||\phi_k||$")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Segment {segment_idx} · mode {k + 1}")
    for k in range(n_show, nrows * ncols):
        axs[k // ncols][k % ncols].axis("off")
    fig2.suptitle("POD spatial modes (L2 norm over channels, XY projection)", fontsize=12, y=1.02)
    fig2.tight_layout()
    fig2.savefig(f"{out_prefix}_segment{segment_idx}_modes.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)


def plot_relative_l2_per_frame(
    pred: np.ndarray,
    true: np.ndarray,
    time_axis: Optional[np.ndarray],
    out_path: str,
    title: str = "Relative L2 error per frame",
) -> np.ndarray:
    """
    pred, true: (T, N, C)
    Per-frame relative error ||pred_t - true_t|| / (||true_t|| + eps); returns rel_l2 (T,).
    """
    assert pred.shape == true.shape
    t_total = pred.shape[0]
    rel = np.empty(t_total, dtype=np.float64)
    eps = 1e-12
    for t in range(t_total):
        diff = pred[t].ravel() - true[t].ravel()
        num = float(np.linalg.norm(diff))
        den = float(np.linalg.norm(true[t].ravel()))
        rel[t] = num / (den + eps)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(t_total) if time_axis is None or len(time_axis) != t_total else time_axis
    ax.plot(x, rel, "b.-", markersize=3)
    ax.set_xlabel("Frame index" if (time_axis is None or len(time_axis) != t_total) else "Time")
    ax.set_ylabel(r"$\| \hat{u}-u\|_2 / \|u\|_2$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return rel


def compute_per_frame_rmse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Root mean square error over all nodes and channels at each time."""
    t_total = pred.shape[0]
    out = np.empty(t_total, dtype=np.float64)
    for t in range(t_total):
        out[t] = float(np.sqrt(np.mean((pred[t] - true[t]) ** 2)))
    return out


def pod_subspace_relative_error(
    train_seq: np.ndarray, frames: np.ndarray, energy_ratio: float
) -> np.ndarray:
    """
    Relative L2 error ‖u - P(u)‖ / ‖u‖ where P is the POD projector fitted on train_seq only.
    Lower bound for any model that stays in that POD subspace when coefficients are exact.
    If this is already large on test frames, the snapshot subspace does not capture crushing motion.
    """
    reducer = PODReducer(energy_ratio)
    reducer.fit(train_seq)
    rel = np.empty(frames.shape[0], dtype=np.float64)
    for t in range(frames.shape[0]):
        snap = frames[t : t + 1]
        coeff = reducer.transform(snap)
        rec = reducer.inverse_transform(coeff)[0]
        num = float(np.linalg.norm((rec - frames[t]).ravel()))
        den = float(np.linalg.norm(frames[t].ravel()) + 1e-12)
        rel[t] = num / den
    return rel


def run_forecast_error_analysis(
    pred: np.ndarray,
    true: np.ndarray,
    coords_xyz: np.ndarray,
    time_axis: np.ndarray,
    train_seq: np.ndarray,
    energy_ratio: float,
    out_dir: str,
    file_prefix: str,
    scatter_max: int,
    seed: int,
) -> None:
    """
    Diagnose where forecast error is large: time, space (XY), channel, vs POD train-subspace ceiling.
    """
    assert pred.shape == true.shape
    n, c = pred.shape[1], pred.shape[2]
    idx_ic = 0
    idx_fc = slice(1, None)
    pred_f = pred[idx_fc]
    true_f = true[idx_fc]
    n_fc = pred_f.shape[0]
    if n_fc == 0:
        print("[DIAG] No forecast frames; skip error analysis.")
        return

    eps = 1e-12
    rel_all = np.array(
        [
            float(np.linalg.norm((pred[t] - true[t]).ravel()))
            / (float(np.linalg.norm(true[t].ravel())) + eps)
            for t in range(pred.shape[0])
        ],
        dtype=np.float64,
    )
    rmse_t = compute_per_frame_rmse(pred, true)

    rel_ic = rel_all[idx_ic]
    rel_forecast = rel_all[idx_fc]

    pod_rel_f = pod_subspace_relative_error(train_seq, true_f, energy_ratio)

    # Spatial: time-mean and time-max of pointwise vector error norm (forecast only)
    err_mag = np.linalg.norm(pred_f - true_f, axis=-1)  # (T_fc, N)
    mean_err_node = np.mean(err_mag, axis=0)
    max_err_node = np.max(err_mag, axis=0)
    worst_fc = int(np.argmax(rel_forecast)) if n_fc else 0

    # Per-channel RMSE (forecast)
    ch_rmse = np.sqrt(np.mean((pred_f - true_f) ** 2, axis=(0, 1)))

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, file_prefix)

    print("\n[DIAG] --- Forecast error analysis ---")
    print(f"[DIAG] IC frame (t=0) relative L2: {rel_ic:.6e} (should be ~0; not zero means IC/basis mismatch)")
    if n_fc:
        print(
            f"[DIAG] Forecast frames only: rel L2 mean={rel_forecast.mean():.6e}, "
            f"median={np.median(rel_forecast):.6e}, max={rel_forecast.max():.6e}"
        )
        print(
            f"[DIAG] POD(train) subspace misfit on TRUE test frames (oracle projection): "
            f"mean={pod_rel_f.mean():.6e}, max={pod_rel_f.max():.6e}"
        )
        print(
            f"[DIAG] If POD misfit ≈ forecast error, dynamics/LSTM is not the main limit; "
            f"if forecast >> POD misfit, rollout/LSTM or segment basis switch dominates."
        )
        print(
            f"[DIAG] Worst forecast frame by relative L2: index {worst_fc} in test "
            f"(0..{n_fc - 1}), rel={rel_forecast[worst_fc]:.6e}"
        )
        print(f"[DIAG] Per-channel RMSE (forecast): {ch_rmse}")
        q90, q99 = np.percentile(mean_err_node, [90, 99])
        print(
            f"[DIAG] Time-mean |error| per node: p90={q90:.4e}, p99={q99:.4e}, "
            f"max={mean_err_node.max():.4e} (spatial hotspots)"
        )

    # Time figure: relative L2 + RMSE + POD ceiling
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    x = np.arange(pred.shape[0]) if time_axis is None or len(time_axis) != pred.shape[0] else time_axis
    axes[0].plot(x, rel_all, "b.-", markersize=3, label="ROM forecast")
    if n_fc:
        x_f = x[1:] if len(x) == pred.shape[0] else x[1:]
        if len(x_f) == len(pod_rel_f):
            axes[0].plot(x_f, pod_rel_f, "g--", linewidth=1.5, label="POD(train) oracle on truth")
    _x0 = float(np.asarray(x).reshape(-1)[0]) if np.size(x) else 0.0
    axes[0].axvline(_x0, color="k", linestyle=":", alpha=0.5, label="IC frame")
    axes[0].set_ylabel(r"$\| \hat{u}-u\|_2 / \|u\|_2$")
    axes[0].set_title("Relative L2 per frame (solid=ROM; dashed=POD subspace on true test)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, rmse_t, "r.-", markersize=3)
    axes[1].set_ylabel("RMSE (all nodes)")
    axes[1].set_xlabel("Time" if (time_axis is not None and len(time_axis) == pred.shape[0]) else "Frame index")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{base}_diagnostics_time.png", dpi=150)
    plt.close(fig)

    # Spatial maps (XY)
    xyz = np.asarray(coords_xyz, dtype=np.float64)
    if xyz.shape[0] == n:
        idx = _subsample_indices(n, scatter_max, seed)
        xs, ys = xyz[idx, 0], xyz[idx, 1]

        fig2, ax = plt.subplots(figsize=(7, 5.5))
        sc = ax.scatter(xs, ys, c=mean_err_node[idx], s=2, cmap="inferno", rasterized=True)
        plt.colorbar(sc, ax=ax, label="Time-mean |pred-true| (forecast)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Spatial error hotspot (mean over forecast times; battery crush: check contact band)")
        fig2.tight_layout()
        fig2.savefig(f"{base}_diagnostics_spatial_mean_err.png", dpi=150)
        plt.close(fig2)

        fig3, ax = plt.subplots(figsize=(7, 5.5))
        err_worst = np.linalg.norm(pred_f[worst_fc] - true_f[worst_fc], axis=-1)
        sc = ax.scatter(xs, ys, c=err_worst[idx], s=2, cmap="inferno", rasterized=True)
        plt.colorbar(sc, ax=ax, label="|pred-true|")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Pointwise error at worst forecast frame (fc idx {worst_fc})")
        fig3.tight_layout()
        fig3.savefig(f"{base}_diagnostics_spatial_worst_frame.png", dpi=150)
        plt.close(fig3)
    else:
        print(f"[DIAG] Skip spatial maps: coords {xyz.shape[0]} != nodes {n}")

    # Channel bars
    if c >= 1:
        fig4, ax = plt.subplots(figsize=(6, 3.5))
        labels = ["ux", "uy", "uz"] if c == 3 else [f"ch{k}" for k in range(c)]
        ax.bar(labels, ch_rmse, color="steelblue")
        ax.set_ylabel("RMSE (forecast)")
        ax.set_title("Per-channel RMSE (displacement: often ux,uy,uz)")
        fig4.tight_layout()
        fig4.savefig(f"{base}_diagnostics_channels.png", dpi=150)
        plt.close(fig4)

    np.savez_compressed(
        f"{base}_diagnostics_metrics.npz",
        relative_l2_all=rel_all,
        rmse_per_frame=rmse_t,
        relative_l2_forecast_only=rel_forecast,
        pod_subspace_rel_on_true_test=pod_rel_f,
        mean_err_node=mean_err_node,
        max_err_node=max_err_node,
        worst_forecast_index=worst_fc,
        channel_rmse_forecast=ch_rmse,
    )
    print(f"[DIAG] Saved: {base}_diagnostics_*.png and {base}_diagnostics_metrics.npz\n")


def build_time_split_indices(
    t_total: int,
    train_ratio: float,
    mode: str,
    seq_len: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Build train/test frame indices (chronological order within each set).

    - chronological: first floor(train_ratio*T) frames train, remainder test (suffix test).
    - random_block: random contiguous test block strictly inside [0, T) (not suffix),
      train = prefix ∪ suffix concatenated in time order. IC for rollout = frame start-1.

    Returns (train_idx, test_idx, ic_idx, test_block_start).
    """
    if t_total < 3:
        raise ValueError(f"Need at least 3 time frames, got {t_total}")

    if mode == "chronological":
        train_t = max(int(train_ratio * t_total), seq_len + 2)
        train_t = min(train_t, t_total - 1)
        train_idx = np.arange(train_t, dtype=np.int64)
        test_idx = np.arange(train_t, t_total, dtype=np.int64)
        ic_idx = train_t - 1
        return train_idx, test_idx, ic_idx, int(train_t)

    if mode != "random_block":
        raise ValueError(f"Unknown time_split mode: {mode}")

    n_test = int(round((1.0 - float(train_ratio)) * t_total))
    n_test = max(1, n_test)
    min_train = seq_len + 2
    max_n_test = max(1, t_total - min_train)
    n_test = min(n_test, max_n_test)

    # Need start ∈ [1, T - n_test) so test is not the suffix and ic_idx = start-1 ≥ 0 exists.
    high_excl = t_total - n_test
    if high_excl <= 1:
        print(
            f"[WARN] time_split=random_block: T={t_total}, n_test={n_test} leaves no valid inner block; "
            f"falling back to chronological split."
        )
        train_t = max(int(train_ratio * t_total), seq_len + 2)
        train_t = min(train_t, t_total - 1)
        train_idx = np.arange(train_t, dtype=np.int64)
        test_idx = np.arange(train_t, t_total, dtype=np.int64)
        ic_idx = train_t - 1
        return train_idx, test_idx, ic_idx, int(train_t)

    start = int(rng.integers(1, high_excl))
    test_idx = np.arange(start, start + n_test, dtype=np.int64)
    train_idx = np.concatenate(
        [np.arange(0, start, dtype=np.int64), np.arange(start + n_test, t_total, dtype=np.int64)]
    )
    ic_idx = start - 1
    return train_idx, test_idx, ic_idx, int(start)


def apply_rom_preset(args: argparse.Namespace) -> None:
    """
    Bundled hyperparameters for battery crush / strong spatial-gradient cases:
    delta-coeff LSTM, rollout auxiliary loss, 3D y-first partition, global features, longer history.
    """
    p = str(getattr(args, "rom_preset", "none") or "none").lower().strip()
    if p in ("", "none"):
        return
    if p != "crush":
        raise ValueError(f"Unknown --rom_preset={getattr(args, 'rom_preset', None)!r} (use none or crush)")
    print(
        "[INFO] --rom_preset crush: applying bundled settings — "
        "Δcoeff, rollout H=4 + step_w=ramp + tf linear 1.0→0.6, EMA decay=0.998, coeff SmoothL1, "
        "seq_len=6, hidden=64×2, spatial_grid=2,2,2 order=yxz, global+physical_time+time_feature, "
        "field_loss=0.5, disp channel_w=1,3,3 / stress: per-ch POD std + channelwise POD (7 SVDs) + cap + pod_full_rank + 7ch w, "
        "min_segment_len>=14, merge_short>=22, epochs>=800, weight_decay>=1e-4, grad_clip=1."
    )
    args.lstm_predict_delta = True
    args.use_global_feat = True
    args.use_physical_time_seg = True
    args.use_time_feature = True
    args.train_rollout_horizon = 4
    args.rollout_loss_weight = 0.35
    args.teacher_forcing_ratio = 0.6
    args.tf_ratio_start = 1.0
    args.rollout_step_weights = "ramp"
    args.rollout_sample_prob = 0.6
    args.seq_len = 6
    args.hidden = 64
    args.layers = 2
    args.lstm_coeff_smoothl1 = True
    args.lstm_ema_decay = 0.998
    args.epochs = max(int(args.epochs), 800)
    args.lstm_field_loss_ratio = 0.5
    tgt = str(getattr(args, "target", "displacement"))
    if tgt == "displacement":
        args.lstm_channel_weights = "1,3,3"
    elif tgt == "stress":
        # sigma_xx,yy,zz, tau_xy,yz,xz, sigma_eq — slightly up-weight shears + von Mises
        args.lstm_channel_weights = "1,1,1,1.5,1.5,1.5,2.5"
        args.stress_per_channel_std = True
        args.pod_full_rank = True
        args.stress_pod_channelwise = True
        cap_u = int(getattr(args, "stress_pod_channelwise_cap", 0))
        # Default cap keeps total LSTM width ~7×18; raise (e.g. 40) or use a large value to relax.
        args.stress_pod_channelwise_cap = 18 if cap_u <= 0 else cap_u
    args.weight_decay = max(float(getattr(args, "weight_decay", 0.0)), 1e-4)
    if float(getattr(args, "grad_clip", 0.0)) <= 0.0:
        args.grad_clip = 1.0
    args.min_segment_len = max(int(args.min_segment_len), 14)
    args.merge_short_segments_len = max(int(getattr(args, "merge_short_segments_len", 0)), 22)
    if args.spatial_grid is None or str(args.spatial_grid).strip() == "":
        args.spatial_grid = "2,2,2"
    args.spatial_partition_order = "yxz"
    args.lr_scheduler = "none"


def run(args):
    apply_rom_preset(args)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] Device: {device}")
    if str(getattr(args, "target", "")) == "stress" and str(
        getattr(args, "rom_preset", "none") or "none"
    ).lower() in ("none", ""):
        print(
            "[INFO] Tip (stress): plain defaults match displacement-tuned baselines and often give rel.error "
            "≫0.5. Prefer: `--rom_preset crush` (same ROM tricks + per-channel stress scaling for POD + 7ch weights), "
            "or add `--stress_per_channel_std` and copy your displacement flags manually."
        )

    if args.solver_npz is not None and args.result_npz is not None:
        solver_path, result_path = args.solver_npz, args.result_npz
    elif args.solver_npz is None and args.result_npz is None:
        solver_path, result_path, _, _ = resolve_data_paths(args.data_root, args.target)
    else:
        raise ValueError("Pass both --solver_npz and --result_npz, or neither (then use --data_root).")

    print(f"[INFO] Solver: {solver_path}")
    print(f"[INFO] Result: {result_path}")

    loader = NPZSampleLoader(solver_path, result_path)
    samples = loader.list_samples()
    print(f"[INFO] Shared sample count: {len(samples)}")
    print(f"[INFO] Sample keys (first 5): {samples[: min(5, len(samples))]}")

    sample_key = args.sample_key or samples[0]
    data = loader.load_sample(sample_key, target=args.target)
    field = data["field"]  # (T, N, C)
    print(f"[INFO] Sample: {sample_key}")
    print(f"[INFO] field ({data['field_key']}) shape: {field.shape}, xyzd shape: {data['xyzd'].shape}, time shape: {data['time'].shape}")

    t_total = field.shape[0]
    rng_split = np.random.default_rng(int(args.seed) + 7919)
    train_idx, test_idx, ic_idx, test_block_start = build_time_split_indices(
        t_total,
        float(args.train_ratio),
        str(args.time_split),
        int(args.seq_len),
        rng_split,
    )
    train_seq = field[train_idx]
    test_seq = field[test_idx]
    print(
        f"[INFO] Time split: {args.time_split}, train_ratio={args.train_ratio:.4f}, "
        f"train frames={len(train_seq)}, test frames={len(test_seq)}"
    )
    if str(args.time_split) == "random_block":
        print(
            f"[INFO]   random test block: frame indices [{test_block_start}, {test_block_start + len(test_seq)}), "
            f"rollout IC index={ic_idx} (one frame before block)"
        )

    time_full = np.asarray(data["time"], dtype=np.float64).reshape(-1)
    time_train = time_full[train_idx]
    t_start = float(time_full[ic_idx])
    if str(args.time_split) == "random_block" and len(test_idx) > 0:
        span = time_full[int(ic_idx) : int(test_idx[-1]) + 1]
        dtr = np.diff(span)
    else:
        dtr = np.diff(time_train)
    dt_pred = float(np.mean(dtr)) if dtr.size else 1.0
    if not np.isfinite(dt_pred) or dt_pred <= 0:
        dt_pred = 1.0
        print("[WARN] Invalid mean dt from training times; using dt=1.0 for forecast time feature")

    global_feat_pass: Optional[np.ndarray] = None
    if args.use_global_feat:
        gf = np.asarray(data["global_feat"], dtype=np.float32).reshape(-1)
        if gf.size > 0:
            global_feat_pass = gf
        else:
            print("[WARN] --use_global_feat set but xyzd has no columns past index 3 (empty global_feat)")
    sdf_full = np.asarray(data["sdf_seq"], dtype=np.float32)
    sdf_train = sdf_full[train_idx]
    sdf_time_varying = bool(data.get("sdf_time_varying", False))
    use_sdf_seg_effective = bool(args.use_sdf_segmentation) and sdf_time_varying
    if args.use_sdf_segmentation and not sdf_time_varying:
        print(
            "[WARN] --use_sdf_segmentation: no time-varying SDF in result NPZ "
            "(keys sdf/sdf_time/...); static sdf → SDF segmentation disabled; "
            "using displacement-based breakpoints (add --use_physical_time_seg for ||Δu||/dt weighting)."
        )

    print(
        f"[INFO] ROM extras: physical_time_seg={args.use_physical_time_seg}, "
        f"time_feature={args.use_time_feature}, global_feat_dim={global_feat_pass.size if global_feat_pass is not None else 0}, "
        f"sdf_segmentation_active={use_sdf_seg_effective} (cli_flag={bool(args.use_sdf_segmentation)})"
    )
    print(
        "[INFO] Temporal segmentation (POD/LSTM segments) is computed only on the training frames above "
        "(concatenated prefix+suffix in random_block mode), independent of spatial region partition."
    )

    coords_full = np.asarray(data["xyzd"][:, :3], dtype=np.float32)
    regions: Optional[List[np.ndarray]] = None
    st_rom: Optional[SpatioTemporalROM] = None
    rom: Optional[SegmentedROM] = None
    spatial_meta: Dict[str, object] = {}

    lstm_ch_w = parse_channel_weights(getattr(args, "lstm_channel_weights", None), int(field.shape[2]))
    fit_kw = dict(
        epochs=int(args.epochs),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        weight_decay=float(args.weight_decay),
        lr_scheduler=str(args.lr_scheduler),
        grad_clip=float(args.grad_clip),
        cosine_eta_min=(None if args.cosine_eta_min is None else float(args.cosine_eta_min)),
        lstm_field_loss_ratio=float(getattr(args, "lstm_field_loss_ratio", 0.0)),
        lstm_channel_weights=lstm_ch_w,
        lstm_predict_delta=bool(getattr(args, "lstm_predict_delta", False)),
        train_rollout_horizon=int(getattr(args, "train_rollout_horizon", 1)),
        teacher_forcing_ratio=float(getattr(args, "teacher_forcing_ratio", 1.0)),
        rollout_loss_weight=float(getattr(args, "rollout_loss_weight", 0.0)),
        rollout_sample_prob=float(getattr(args, "rollout_sample_prob", 0.5)),
        rollout_aux_seed=int(args.seed),
        tf_ratio_start=(
            float(args.tf_ratio_start) if getattr(args, "tf_ratio_start", None) is not None else None
        ),
        rollout_step_weights=getattr(args, "rollout_step_weights", None),
        lstm_coeff_smoothl1=bool(getattr(args, "lstm_coeff_smoothl1", False)),
        lstm_ema_decay=float(getattr(args, "lstm_ema_decay", 0.0)),
        merge_short_segments_len=int(getattr(args, "merge_short_segments_len", 0)),
        stress_per_channel_std=bool(getattr(args, "stress_per_channel_std", False)),
        stress_pod_channelwise=bool(getattr(args, "stress_pod_channelwise", False)),
        stress_pod_channelwise_cap=int(getattr(args, "stress_pod_channelwise_cap", 0)),
    )

    use_spatial_3d = args.spatial_grid is not None and str(args.spatial_grid).strip() != ""
    use_spatial_1d = (not use_spatial_3d) and args.spatial_regions > 1
    use_spatial = use_spatial_3d or use_spatial_1d

    if use_spatial_3d:
        gx, gy, gz = parse_spatial_grid(str(args.spatial_grid))
        gx, gy, gz = max(1, gx), max(1, gy), max(1, gz)
        order = _normalize_partition_order(args.spatial_partition_order)
        regions = spatial_partition_3d_nested(
            np.asarray(coords_full, dtype=np.float64), gx, gy, gz, order=order
        )
        spatial_meta = {
            "type": "3d",
            "grid": (gx, gy, gz),
            "order": order,
            "n_regions": len(regions),
        }
        print(
            f"[INFO] Spatio-temporal ROM (3D nested): grid gx,gy,gz=({gx},{gy},{gz}), "
            f"order={order} → {len(regions)} non-empty regions (~equal nodes per cut; "
            f"interfaces not coupled between regions)"
        )
        union = np.sort(np.unique(np.concatenate(regions)))
        if union.size != field.shape[1] or int(union[0]) != 0 or int(union[-1]) != field.shape[1] - 1:
            print(
                f"[WARN] Region cover union size={union.size} vs N={field.shape[1]}; check partitioning"
            )
        for r, idx in enumerate(regions):
            cx, cy, cz = coords_full[idx, 0], coords_full[idx, 1], coords_full[idx, 2]
            print(
                f"[INFO]   Region {r}: |nodes|={idx.size}, "
                f"x[{float(cx.min()):.4f},{float(cx.max()):.4f}] "
                f"y[{float(cy.min()):.4f},{float(cy.max()):.4f}] "
                f"z[{float(cz.min()):.4f},{float(cz.max()):.4f}]"
            )
        st_rom = SpatioTemporalROM(
            region_indices=regions,
            energy_ratio=args.energy_ratio,
            seq_len=args.seq_len,
            lstm_hidden=args.hidden,
            lstm_layers=args.layers,
            threshold_ratio=args.threshold_ratio,
            min_segment_len=args.min_segment_len,
            normalize_coeffs=not args.no_coeff_norm,
            use_physical_time_seg=args.use_physical_time_seg,
            use_time_feature=args.use_time_feature,
            use_sdf_segmentation=use_sdf_seg_effective,
            pod_full_rank=bool(getattr(args, "pod_full_rank", False)),
        )
        st_rom.fit(
            train_seq,
            time_train,
            global_feat_pass,
            sdf_train,
            device=device,
            **fit_kw,
        )
    elif use_spatial_1d:
        regions = spatial_partition_equal_bins(
            coords_full, args.spatial_axis, args.spatial_regions, args.seed
        )
        axis_name = ("x", "y", "z")[args.spatial_axis]
        spatial_meta = {"type": "1d", "axis": args.spatial_axis, "n_bands": args.spatial_regions}
        print(
            f"[INFO] Spatio-temporal ROM (1D): {args.spatial_regions} bands along {axis_name} "
            f"(~equal nodes/band; use --spatial_grid gx,gy,gz for full 3D splitting)"
        )
        union = np.sort(np.unique(np.concatenate(regions)))
        if union.size != field.shape[1] or int(union[0]) != 0 or int(union[-1]) != field.shape[1] - 1:
            print(
                f"[WARN] Region cover union size={union.size} vs N={field.shape[1]}; check partitioning"
            )
        for r, idx in enumerate(regions):
            cband = coords_full[idx, args.spatial_axis]
            print(
                f"[INFO]   Region {r}: |nodes|={idx.size}, {axis_name} in "
                f"[{float(cband.min()):.4f}, {float(cband.max()):.4f}]"
            )
        st_rom = SpatioTemporalROM(
            region_indices=regions,
            energy_ratio=args.energy_ratio,
            seq_len=args.seq_len,
            lstm_hidden=args.hidden,
            lstm_layers=args.layers,
            threshold_ratio=args.threshold_ratio,
            min_segment_len=args.min_segment_len,
            normalize_coeffs=not args.no_coeff_norm,
            use_physical_time_seg=args.use_physical_time_seg,
            use_time_feature=args.use_time_feature,
            use_sdf_segmentation=use_sdf_seg_effective,
            pod_full_rank=bool(getattr(args, "pod_full_rank", False)),
        )
        st_rom.fit(
            train_seq,
            time_train,
            global_feat_pass,
            sdf_train,
            device=device,
            **fit_kw,
        )
    else:
        rom = SegmentedROM(
            energy_ratio=args.energy_ratio,
            seq_len=args.seq_len,
            lstm_hidden=args.hidden,
            lstm_layers=args.layers,
            threshold_ratio=args.threshold_ratio,
            min_segment_len=args.min_segment_len,
            normalize_coeffs=not args.no_coeff_norm,
            use_physical_time_seg=args.use_physical_time_seg,
            use_time_feature=args.use_time_feature,
            use_sdf_segmentation=use_sdf_seg_effective,
            pod_full_rank=bool(getattr(args, "pod_full_rank", False)),
        )
        rom.fit(
            train_seq,
            time_train,
            global_feat_pass,
            device=device,
            sdf_seq=sdf_train,
            **fit_kw,
        )

    os.makedirs(args.out_dir, exist_ok=True)
    plot_prefix = os.path.join(args.out_dir, f"pod_{sample_key}_{args.target}")
    pod_ch_names: Optional[List[str]] = list(TARGET_CONFIG[str(args.target)]["components"])

    if not args.no_pod_plots:
        if use_spatial and st_rom is not None and regions is not None:
            for r, idx in enumerate(regions):
                crd = coords_full[idx]
                rpfx = os.path.join(args.out_dir, f"pod_{sample_key}_{args.target}_reg{r}")
                for seg_i, seg_model in enumerate(st_rom.roms[r].models):
                    plot_pod_segment_analysis(
                        seg_model.reducer,
                        segment_idx=seg_i,
                        coords_xyz=crd,
                        out_prefix=rpfx,
                        n_modes_spatial=args.pod_modes_plot,
                        scatter_max_points=args.pod_scatter_max,
                        subsample_seed=args.seed,
                        channel_names=pod_ch_names,
                    )
            print(f"[INFO] POD spectrum/mode figures saved: {plot_prefix}_reg*_segment*.png")
        elif rom is not None:
            for seg_i, seg_model in enumerate(rom.models):
                plot_pod_segment_analysis(
                    seg_model.reducer,
                    segment_idx=seg_i,
                    coords_xyz=coords_full,
                    out_prefix=plot_prefix,
                    n_modes_spatial=args.pod_modes_plot,
                    scatter_max_points=args.pod_scatter_max,
                    subsample_seed=args.seed,
                    channel_names=pod_ch_names,
                )
            print(f"[INFO] POD spectrum/mode figures saved: {plot_prefix}_segment*.png")

    initial = field[ic_idx]
    n_pred = len(test_seq)
    if use_spatial and st_rom is not None:
        pred = st_rom.predict(
            initial, n_pred, device=device, t_start=t_start, dt=dt_pred
        )
    elif rom is not None:
        pred = rom.predict(
            initial, n_pred, device=device, t_start=t_start, dt=dt_pred
        )
    else:
        raise RuntimeError("No ROM fitted")
    true = np.concatenate([field[ic_idx : ic_idx + 1], test_seq], axis=0)

    rmse = np.sqrt(np.mean((pred - true) ** 2))
    rel = np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12)
    print(f"[RESULT] RMSE={rmse:.6e}, RelativeL2={rel:.6e}")

    time_pred = np.concatenate(
        [time_full[ic_idx : ic_idx + 1], time_full[test_idx]],
        axis=0,
    )
    rel_per_frame = None
    if not args.no_rel_l2_plot:
        rel_path = os.path.join(args.out_dir, f"rel_l2_{sample_key}_{args.target}.png")
        rel_per_frame = plot_relative_l2_per_frame(
            pred,
            true,
            time_axis=time_pred,
            out_path=rel_path,
            title=f"{sample_key} · {args.target} · relative L2 (forecast)",
        )
        print(f"[INFO] Per-frame relative L2 plot: {rel_path}")
        print(f"[RESULT] Per-frame relative L2: mean={rel_per_frame.mean():.4e}, max={rel_per_frame.max():.4e}")

    out_npz = os.path.join(args.out_dir, f"pred_{sample_key}_{args.target}.npz")
    save_kw = dict(
        sample_key=sample_key,
        target=args.target,
        target_field=args.target,
        field_key=data["field_key"],
        pred=pred,
        true=true,
        time=time_pred,
        rmse=rmse,
        relative_l2=rel,
        spatial_regions=int(args.spatial_regions),
        spatial_axis=int(args.spatial_axis),
        use_physical_time_seg=bool(args.use_physical_time_seg),
        use_time_feature=bool(args.use_time_feature),
        use_global_feat=bool(args.use_global_feat),
        use_sdf_segmentation=bool(args.use_sdf_segmentation),
        use_sdf_segmentation_active=bool(use_sdf_seg_effective),
        sdf_time_varying=bool(data.get("sdf_time_varying", False)),
        forecast_t_start=np.float64(t_start),
        forecast_dt=np.float64(dt_pred),
        time_split_mode=np.array([str(args.time_split)], dtype=object),
        train_frame_indices=train_idx.astype(np.int64),
        test_frame_indices=test_idx.astype(np.int64),
        ic_frame_index=np.int64(ic_idx),
        rom_preset=np.array([str(getattr(args, "rom_preset", "none"))], dtype=object),
    )
    ptype = str(spatial_meta.get("type", "none"))
    save_kw["spatial_partition_type"] = np.array([ptype], dtype=object)
    if spatial_meta.get("type") == "3d":
        g3 = spatial_meta["grid"]
        save_kw["spatial_grid_gx_gy_gz"] = np.array([g3[0], g3[1], g3[2]], dtype=np.int32)
        save_kw["spatial_partition_order_str"] = np.array([str(spatial_meta["order"])], dtype=object)
    if use_spatial and regions is not None:
        save_kw["region_node_counts"] = np.array([len(ix) for ix in regions], dtype=np.int64)
    if rel_per_frame is not None:
        save_kw["relative_l2_per_frame"] = rel_per_frame
    np.savez_compressed(out_npz, **save_kw)
    print(f"[INFO] Predictions saved: {out_npz}")

    if not args.no_error_analysis:
        safe_prefix = f"errors_{sample_key}_{args.target}".replace(" ", "_")
        run_forecast_error_analysis(
            pred=pred,
            true=true,
            coords_xyz=data["xyzd"][:, :3],
            time_axis=time_pred,
            train_seq=train_seq,
            energy_ratio=args.energy_ratio,
            out_dir=args.out_dir,
            file_prefix=safe_prefix,
            scatter_max=args.error_scatter_max,
            seed=args.seed,
        )


def build_parser():
    epilog = """
Preset (crush / contact-band, targets rollout gap + stable coeff training):
  python segmented_rom_pytorch.py --data_root DIR --rom_preset crush
  Includes: delta-coeff LSTM, rollout H=4 with step_w=ramp, tf linear 1.0→0.6, rollout_w≈0.35,
  coeff SmoothL1, EMA decay 0.998 on weights for inference, field_loss 0.5, channel_w 1,3,3 (disp),
  seq_len=6, hidden 64 ×2 layers, spatial_grid 2,2,2 yxz, min_segment_len≥14, merge_short_segments≥22, epochs≥800, etc.
  For --target stress, crush also enables channelwise POD (7 SVDs), pod_full_rank, per-channel std scaling, and
  stress_pod_channelwise_cap=18 unless you set --stress_pod_channelwise_cap.
  Note: preset applies inside run() (overwrites). For a custom mix use --rom_preset none.
"""
    parser = argparse.ArgumentParser(
        "Segmented POD + LSTM ROM (paths aligned with read_and_visualize_dataset.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument(
        "--rom_preset",
        type=str,
        default="none",
        choices=["none", "crush"],
        help="crush: apply bundled crush-ROM hyperparameters inside run() (see epilog below). "
        "Applies in run() after parse (overwrites those hyperparameters). Use none + manual flags to customize.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=".",
        help="Dataset root; expects DataSet_*/SolverFileData.npz like the visualization script",
    )
    parser.add_argument(
        "--solver_npz",
        type=str,
        default=None,
        help="Optional full path to SolverFileData.npz (requires --result_npz)",
    )
    parser.add_argument(
        "--result_npz",
        type=str,
        default=None,
        help="Optional full path to ResultFileData_displacement.npz or ResultFileData_stress.npz",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="displacement",
        choices=["displacement", "stress"],
        help="Same as read_and_visualize_dataset --target",
    )
    parser.add_argument(
        "--sample_key",
        type=str,
        default=None,
        help="Sample key, e.g. Sample_0; default: first shared key after sort",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Fraction of frames used for training (test size ≈ 1 - ratio); default 0.9 → ~9:1 train:test",
    )
    parser.add_argument(
        "--time_split",
        type=str,
        default="random_block",
        choices=["chronological", "random_block"],
        help="chronological: first frames train, last frames test. random_block: random contiguous inner test block "
        "(train = prefix ∪ suffix, rollout from frame before block); reproducible via --seed",
    )
    parser.add_argument("--energy_ratio", type=float, default=0.999999)
    parser.add_argument(
        "--pod_full_rank",
        action="store_true",
        help="Retain all non-zero SVD modes up to min(N·C, T) per segment (no energy truncation). "
        "crush+stress turns this on: squeezes extra snapshot-rank modes when T is small.",
    )
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--threshold_ratio", type=float, default=0.5)
    parser.add_argument(
        "--spatial_grid",
        type=str,
        default=None,
        help="3D nested split: gx,gy,gz (e.g. 2,2,2 → up to 8 boxes, ~equal nodes per cut on x,y,z). "
        "Uses full xyz from SolverFileData. If set, overrides --spatial_regions. "
        "Try 2,2,1 with --spatial_partition_order yxz for crush stratified along y then x.",
    )
    parser.add_argument(
        "--spatial_partition_order",
        type=str,
        default="yxz",
        help="Nested axis order, any permutation of xyz (default yxz: split y first, then x, then z).",
    )
    parser.add_argument(
        "--spatial_regions",
        type=int,
        default=1,
        help="1D-only: if >1 and --spatial_grid unset, split into this many bands along --spatial_axis",
    )
    parser.add_argument(
        "--spatial_axis",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="1D-only: axis for --spatial_regions (0=x,1=y,2=z)",
    )
    parser.add_argument(
        "--min_segment_len",
        type=int,
        default=5,
        help="Min frames per temporal segment; with --use_physical_time_seg many breakpoints → short segments "
        "and tiny n_win → LSTM overfits. Prefer 12–25 when using physical-time segmentation.",
    )
    parser.add_argument(
        "--no_coeff_norm",
        action="store_true",
        help="Disable per-segment standardization of POD coefficients before LSTM (not recommended)",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="none",
        choices=["none", "cosine"],
        help="cosine: CosineAnnealingLR (eta_min defaults to max(1e-8, 0.05*lr), not 0 — avoids dead training at end). "
        "Risky when n_win is tiny (see WARN).",
    )
    parser.add_argument(
        "--cosine_eta_min",
        type=float,
        default=None,
        help="If set with --lr_scheduler cosine, minimum LR (absolute). Default: auto max(1e-8, 0.05*--lr)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Adam L2 penalty on LSTM weights (e.g. 1e-5); 0 disables",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=0.0,
        help="If >0, clip LSTM grad norm (e.g. 1.0); 0 disables",
    )
    parser.add_argument(
        "--lstm_field_loss_ratio",
        type=float,
        default=0.0,
        help="Blend LSTM training loss: (1-a)*MSE(normalized coeffs) + a * weighted MSE in physical field "
        "(POD inverse). Use 0.3–0.7 when uy/uz (or some stress comps) dominate forecast error. 0=off.",
    )
    parser.add_argument(
        "--lstm_channel_weights",
        type=str,
        default=None,
        help="Comma-separated weights per field component for field loss (displacement: ux,uy,uz). "
        "Example 1,3,3 up-weights y,z vs x. One number repeats to all channels. Default: uniform.",
    )
    parser.add_argument(
        "--lstm_predict_delta",
        action="store_true",
        help="Train LSTM on Δcoeff (normalized) and rollout with accumulation; often smoother than absolute coeff.",
    )
    parser.add_argument(
        "--train_rollout_horizon",
        type=int,
        default=1,
        help="If >=2, add random multi-step rollout auxiliary loss (teacher forcing) to reduce train/inference gap. "
        "Use with --rollout_loss_weight > 0.",
    )
    parser.add_argument(
        "--teacher_forcing_ratio",
        type=float,
        default=1.0,
        help="In rollout auxiliary loss: probability of feeding true next coeff into history (1=pure teacher forcing).",
    )
    parser.add_argument(
        "--rollout_loss_weight",
        type=float,
        default=0.0,
        help="Weight for multi-step rollout MSE (see --train_rollout_horizon). Try 0.2–0.6 with horizon 3–5.",
    )
    parser.add_argument(
        "--rollout_sample_prob",
        type=float,
        default=0.5,
        help="Per batch, probability to include rollout auxiliary loss (reduces overhead when horizon>1).",
    )
    parser.add_argument(
        "--tf_ratio_start",
        type=float,
        default=None,
        help="If set, teacher-forcing ratio in rollout loss linearly anneals from this value to "
        "--teacher_forcing_ratio over training epochs (early: more true history, late: more own preds).",
    )
    parser.add_argument(
        "--rollout_step_weights",
        type=str,
        default=None,
        help="Comma-separated H weights for rollout MSE (H=--train_rollout_horizon), or 'ramp' (1..H), "
        "or 'late' (emphasize far steps). Default: uniform.",
    )
    parser.add_argument(
        "--lstm_coeff_smoothl1",
        action="store_true",
        help="Use SmoothL1 for coefficient (and rollout coeff) loss; often more stable than pure MSE.",
    )
    parser.add_argument(
        "--lstm_ema_decay",
        type=float,
        default=0.0,
        help="If in (0,1), maintain EMA of LSTM weights during training and load EMA into model for predict "
        "(e.g. 0.998). 0 disables.",
    )
    parser.add_argument(
        "--merge_short_segments_len",
        type=int,
        default=0,
        help="If >1, merge adjacent temporal segments until each has at least this many frames (reduces n_win≈6 "
        "overfitting on tail segments). Try 20–28 with seq_len=6. 0=off. crush preset sets ≥22.",
    )
    parser.add_argument(
        "--stress_per_channel_std",
        action="store_true",
        help="For target=stress (7 components): scale each stress channel by its std over the segment before POD, "
        "then unscale after inverse (balances σ vs τ in the basis). On with --rom_preset crush for stress.",
    )
    parser.add_argument(
        "--stress_pod_channelwise",
        action="store_true",
        help="Stress only (7 ch): separate POD/SVD per scalar component on (N,T), concatenate coeffs. "
        "Reduces cross-component mixing in spatial modes. crush+stress enables by default.",
    )
    parser.add_argument(
        "--stress_pod_channelwise_cap",
        type=int,
        default=0,
        help="Max POD modes per stress component when --stress_pod_channelwise (0=no cap). "
        "crush+stress sets 18 if you leave 0; use a large value (e.g. 80) to lift the cap.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="outputs_segmented_rom")
    parser.add_argument("--no_pod_plots", action="store_true", help="Skip POD spectrum and spatial mode figures")
    parser.add_argument("--no_rel_l2_plot", action="store_true", help="Skip per-frame relative L2 plot")
    parser.add_argument(
        "--pod_modes_plot",
        type=int,
        default=6,
        help="Number of spatial POD modes to plot per segment (scatter)",
    )
    parser.add_argument(
        "--pod_scatter_max",
        type=int,
        default=40000,
        help="Max nodes in mode scatter (random subsample if larger)",
    )
    parser.add_argument(
        "--no_error_analysis",
        action="store_true",
        help="Skip forecast error diagnostics (time/space/channel + POD subspace ceiling)",
    )
    parser.add_argument(
        "--error_scatter_max",
        type=int,
        default=40000,
        help="Max nodes in diagnostic spatial scatter plots",
    )
    parser.add_argument(
        "--use_physical_time_seg",
        action="store_true",
        help="Weight temporal segmentation by ||Δu||/(sqrt(N)*dt) using physical time from solver NPZ",
    )
    parser.add_argument(
        "--use_time_feature",
        action="store_true",
        help="Concatenate per-frame time (normalized within each temporal segment) to LSTM input; forecast uses mean train dt",
    )
    parser.add_argument(
        "--use_global_feat",
        action="store_true",
        help="Concatenate xyzd[0,4:] (sample-constant global features) to each LSTM input row",
    )
    parser.add_argument(
        "--use_sdf_segmentation",
        action="store_true",
        help="Temporal breakpoints from mean(|Δsdf|)/dt (per-node SDF from result NPZ if present; else static sdf → no temporal change)",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)

# Displacement error analysis:
# [DIAG] --- Forecast error analysis ---
# [DIAG] IC frame (t=0) relative L2: 0.000000e+00 (should be ~0; not zero means IC/basis mismatch)
# [DIAG] Forecast frames only: rel L2 mean=7.272433e-02, median=7.155647e-02, max=9.792300e-02
# [DIAG] POD(train) subspace misfit on TRUE test frames (oracle projection): mean=1.496799e-03, max=1.870044e-03
# [DIAG] If POD misfit ≈ forecast error, dynamics/LSTM is not the main limit; if forecast >> POD misfit, rollout/LSTM or segment basis switch dominates.
# [DIAG] Worst forecast frame by relative L2: index 4 in test (0..4), rel=9.792300e-02
# [DIAG] Per-channel RMSE (forecast): [2.2169414 7.755486  9.843143 ]
# [DIAG] Time-mean |error| per node: p90=2.0098e+01, p99=2.5930e+01, max=3.2676e+01 (spatial hotspots)
# [DIAG] Saved: outputs_segmented_rom/errors_Sample_0_displacement_diagnostics_*.png and outputs_segmented_rom/errors_Sample_0_displacement_diagnostics_metrics.npz