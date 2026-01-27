from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage import measure

from app.inference import ModelResult


GT_COLOR = "#0CDA3C"
PRED_COLORS = {
    "WT": "#EA41F9",
    "TC": "#DA6675",
    "ET": "#F8F41E",
}
BRAIN_COLOR = "#97b0cf"


def make_roi_masks(seg: np.ndarray) -> dict:
    return {
        "WT": seg > 0,
        "TC": np.isin(seg, [1, 3]),
        "ET": seg == 3,
    }


def compute_bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return mins, maxs


def crop_and_downsample(mask: np.ndarray, step: int, margin: int):
    bbox = compute_bbox(mask)
    if bbox is None:
        return None, None
    (z0, y0, x0), (z1, y1, x1) = bbox
    z0 = max(0, z0 - margin)
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    z1 = min(mask.shape[0], z1 + margin)
    y1 = min(mask.shape[1], y1 + margin)
    x1 = min(mask.shape[2], x1 + margin)
    cropped = mask[z0:z1, y0:y1, x0:x1]
    if step > 1:
        cropped = cropped[::step, ::step, ::step]
    offset = np.array([z0, y0, x0], dtype=np.float32)
    return cropped, offset


def mask_to_mesh_trace(
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    color: str,
    name: str,
    opacity: float,
    step: int,
    margin: int,
) -> Optional[go.Mesh3d]:
    if mask is None or mask.sum() == 0:
        return None
    prepared, offset = crop_and_downsample(mask, step=step, margin=margin)
    if prepared is None or prepared.sum() == 0:
        if step > 1:
            prepared, offset = crop_and_downsample(mask, step=1, margin=margin)
        if prepared is None or prepared.sum() == 0:
            return None
    if step <= 1:
        step_spacing = spacing
    else:
        step_spacing = (spacing[0] * step, spacing[1] * step, spacing[2] * step)
    verts, faces, _, _ = measure.marching_cubes(prepared.astype(np.float32), level=0.5, spacing=step_spacing)
    verts += offset * np.array(spacing)
    return go.Mesh3d(
        x=verts[:, 2],
        y=verts[:, 1],
        z=verts[:, 0],
        i=faces[:, 2],
        j=faces[:, 1],
        k=faces[:, 0],
        color=color,
        opacity=opacity,
        name=name,
        flatshading=True,
        showscale=False,
    )


def build_plotly_figure(
    results: List[ModelResult],
    title: str,
    show_gt_row: bool = False,
    overlay_gt: bool = False,
    show_brain: bool = True,
    sync_camera: bool = True,
    sample_step: int = 2,
    margin: int = 2,
    brain_sample_step: int = 4,
) -> Optional[str]:
    if not results:
        return None

    row_titles: List[str] = []
    added_gt_kinds = set()

    for res in results:
        if show_gt_row and res.kind not in added_gt_kinds and res.gt_seg is not None:
            row_titles.extend([f"GT {res.kind.upper()} - {roi}" for roi in ["WT", "TC", "ET"]])
            added_gt_kinds.add(res.kind)
        row_titles.extend([f"{res.name} - {roi}" for roi in ["WT", "TC", "ET"]])

    rows = len(row_titles) // 3
    specs = [[{"type": "scene"}] * 3 for _ in range(rows)]
    fig = make_subplots(
        rows=rows,
        cols=3,
        specs=specs,
        subplot_titles=row_titles,
        horizontal_spacing=0.03,
        vertical_spacing=0.04,
    )

    def add_row(
        row_idx: int,
        label: str,
        pred_seg: Optional[np.ndarray],
        gt_seg_row: Optional[np.ndarray],
        brain_mask_row: Optional[np.ndarray],
        spacing_row: Tuple[float, float, float],
    ):
        for col_idx, roi in enumerate(["WT", "TC", "ET"], start=1):
            if show_brain and brain_mask_row is not None:
                brain_trace = mask_to_mesh_trace(
                    mask=brain_mask_row,
                    spacing=spacing_row,
                    color=BRAIN_COLOR,
                    name="Brain",
                    opacity=0.15,
                    step=max(1, int(brain_sample_step)),
                    margin=0,
                )
                if brain_trace is not None:
                    fig.add_trace(brain_trace, row=row_idx, col=col_idx)

            if gt_seg_row is not None:
                gt_trace = mask_to_mesh_trace(
                    mask=make_roi_masks(gt_seg_row)[roi],
                    spacing=spacing_row,
                    color=GT_COLOR,
                    name=f"{label} GT {roi}",
                    opacity=0.55,
                    step=max(1, int(sample_step)),
                    margin=max(0, int(margin)),
                )
                if gt_trace is not None:
                    fig.add_trace(gt_trace, row=row_idx, col=col_idx)

            if pred_seg is not None:
                pred_trace = mask_to_mesh_trace(
                    mask=make_roi_masks(pred_seg)[roi],
                    spacing=spacing_row,
                    color=PRED_COLORS.get(roi, "#F916CB"),
                    name=f"{label} Pred {roi}",
                    opacity=0.45,
                    step=max(1, int(sample_step)),
                    margin=max(0, int(margin)),
                )
                if pred_trace is not None:
                    fig.add_trace(pred_trace, row=row_idx, col=col_idx)

    row_cursor = 1
    added_gt_kinds.clear()
    for res in results:
        spacing = res.spacing or (1.0, 1.0, 1.0)
        if show_gt_row and res.kind not in added_gt_kinds and res.gt_seg is not None:
            add_row(
                row_cursor,
                f"GT {res.kind.upper()}",
                None,
                res.gt_seg,
                res.brain_mask,
                spacing,
            )
            row_cursor += 1
            added_gt_kinds.add(res.kind)

        add_row(
            row_cursor,
            res.name,
            res.pred_seg,
            res.gt_seg if overlay_gt else None,
            res.brain_mask,
            spacing,
        )
        row_cursor += 1

    axis_cfg = dict(title="", showticklabels=False, visible=False, showgrid=False, zeroline=False)
    for i in range(1, rows * 3 + 1):
        scene_key = "scene" if i == 1 else f"scene{i}"
        fig.update_layout(
            **{
                scene_key: dict(
                    aspectmode="data",
                    xaxis=axis_cfg,
                    yaxis=axis_cfg,
                    zaxis=axis_cfg,
                    dragmode="turntable",
                    bgcolor="white",
                )
            }
        )

    fig.update_layout(
        height=max(720, rows * 280),
        width=1600,
        title_text=title,
        legend=dict(orientation="h", yanchor="bottom", y=0.01, x=0.5, xanchor="center"),
    )

    return figure_to_html(fig, sync_camera=sync_camera)


def figure_to_html(fig: go.Figure, sync_camera: bool = True) -> str:
    import plotly.io as pio

    div_id = "plotly-figure"
    body = pio.to_html(fig, include_plotlyjs="inline", full_html=False, div_id=div_id)
    sync_flag = "true" if sync_camera else "false"
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    body {{ margin: 0; background: #ffffff; }}
    #plotly-figure {{ width: 100vw; height: 100vh; }}
  </style>
</head>
<body>
  {body}
  <script>
    const syncEnabled = {sync_flag};
    const gd = document.getElementById("{div_id}");
    let isSyncing = false;

    function applyCamera(camera) {{
      if (!syncEnabled || !camera) return;
      const update = {{}};
      const keys = Object.keys(gd._fullLayout || {{}}).filter(k => k.startsWith("scene"));
      keys.forEach(k => {{
        update[k + ".camera"] = camera;
      }});
      if (Object.keys(update).length === 0) return;
      isSyncing = true;
      Plotly.relayout(gd, update).then(() => {{
        isSyncing = false;
      }});
    }}

    gd.on('plotly_relayout', function(e) {{
      if (isSyncing) return;
      const camKey = Object.keys(e).find(k => k.endsWith(".camera"));
      if (camKey) {{
        applyCamera(e[camKey]);
      }}
    }});
  </script>
</body>
</html>
"""
