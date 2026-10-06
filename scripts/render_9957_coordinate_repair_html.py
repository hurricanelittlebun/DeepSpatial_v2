"""Create a standalone, dependency-free interactive HTML for 9957/g0 QC.

The HTML intentionally visualizes the same coordinate objects as the final QC:
ST anchor centroids in the repaired global frame and the valid H&E/UNI2 feature
grid support used by the reconstruction code.  It does not draw a raw mask
through an incomplete transform.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from render_9957_coordinate_repair_qc import _load_feature_records, _downsample  # noqa: E402


DEFAULT_GROUP = ROOT / "data" / "9957_g0"
DEFAULT_CANDIDATE = (
    DEFAULT_GROUP
    / "registration_correction_he_v5_coordinate_repair"
    / "9957_g0__st_to_global_v4_repaired_candidate.h5ad"
)
DEFAULT_FEATURE_STORE = (
    DEFAULT_GROUP
    / "reconstruction_prep_v5_coordinate_repair"
    / "uni2_features_global_v4_repaired.h5"
)
DEFAULT_OUTPUT = (
    DEFAULT_GROUP
    / "qc"
    / "coordinate_repair_v5_final_qc_v2"
    / "coordinate_repair_opaque_high_contrast_v1.html"
)

COLORS = [
    "#e41a1c",
    "#0057b7",
    "#00a651",
    "#6a1b9a",
    "#ff7f00",
    "#00a6d6",
    "#b8860b",
    "#4d4d4d",
    "#e6007e",
    "#006400",
]


def _json_points(points: np.ndarray, max_points: int) -> list[list[float]]:
    sampled = _downsample(np.asarray(points, dtype=np.float64), max_points)
    # Three decimals are far below the visual resolution of this overview,
    # while keeping the standalone HTML substantially smaller.
    return np.round(sampled, 3).tolist()


def _load_payload(candidate: Path, feature_store: Path) -> dict:
    records, frame = _load_feature_records(
        candidate, feature_store, max_boundary_points=8000
    )
    sections = []
    for index, record in enumerate(records):
        sections.append(
            {
                "id": int(record["section_id"]),
                "z_um": float(record["z_um"]),
                "color": COLORS[index % len(COLORS)],
                "n_cells": int(record["n_cells"]),
                "feature_valid_fraction": float(record["feature_valid_fraction"]),
                "quality_1_fraction": float(record["quality_1_fraction"]),
                "quality_2_fraction": float(record["quality_2_fraction"]),
                "quality_3_fraction": float(record["quality_3_fraction"]),
                "st": _json_points(record["points"], max_points=20000),
                "he": _json_points(record["feature_boundary"], max_points=8000),
            }
        )
    total = sum(section["n_cells"] for section in sections)
    weighted = sum(
        section["n_cells"] * section["feature_valid_fraction"]
        for section in sections
    ) / max(total, 1)
    return {
        "coordinate_frame": frame,
        "sections": sections,
        "total_st_cells": int(total),
        "weighted_feature_valid_fraction": float(weighted),
        "sampled_st_points": int(sum(len(section["st"]) for section in sections)),
        "support_boundary_points": int(sum(len(section["he"]) for section in sections)),
    }


def _build_html(payload: dict, output: Path) -> str:
    data_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    frame_label = html.escape(payload["coordinate_frame"])
    generated_label = html.escape(str(output.resolve()))
    coverage_label = f"{100.0 * payload['weighted_feature_valid_fraction']:.2f}%"
    return r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>9957/g0 坐标修正交互式质控</title>
<style>
  :root { color-scheme: light; --ink:#1f2937; --muted:#64748b; --line:#dbe3ee; --panel:#f8fafc; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink); background:#fff; }
  header { padding:18px 24px 10px; border-bottom:1px solid var(--line); }
  h1 { margin:0 0 5px; font-size:22px; }
  h2 { margin:0 0 8px; font-size:17px; }
  p { margin:5px 0; line-height:1.5; }
  .sub { color:var(--muted); font-size:13px; }
  .layout { display:grid; grid-template-columns:minmax(0,1fr) 320px; gap:16px; padding:16px 24px 28px; }
  .card { border:1px solid var(--line); border-radius:10px; background:#fff; padding:12px; box-shadow:0 1px 3px rgba(15,23,42,.05); }
  .plot-wrap { position:relative; width:100%; height:min(78vh,820px); min-height:520px; background:#fff; border:1px solid #cbd5e1; border-radius:7px; overflow:hidden; }
  canvas { width:100%; height:100%; display:block; cursor:grab; }
  canvas.dragging { cursor:grabbing; }
  .plot-caption { display:flex; justify-content:space-between; gap:10px; color:var(--muted); font-size:12px; padding-top:7px; }
  .controls { display:flex; flex-wrap:wrap; gap:7px; margin:8px 0 12px; }
  button { border:1px solid #b9c6d8; background:#fff; color:var(--ink); border-radius:6px; padding:6px 9px; cursor:pointer; }
  button:hover { background:#eef5ff; }
  label.switch { display:inline-flex; align-items:center; gap:5px; font-size:13px; }
  .legend { display:grid; grid-template-columns:1fr 1fr; gap:5px 10px; margin:8px 0; }
  .legend label { display:flex; align-items:center; gap:6px; font-size:12px; white-space:nowrap; }
  .swatch { width:12px; height:12px; border-radius:3px; display:inline-block; border:1px solid rgba(0,0,0,.15); }
  .note { background:var(--panel); border-left:3px solid #3b82f6; padding:9px 10px; font-size:12px; color:#334155; margin:8px 0 12px; }
  .metric { width:100%; border-collapse:collapse; font-size:11px; }
  .metric th,.metric td { border-bottom:1px solid var(--line); padding:5px 4px; text-align:right; }
  .metric th:first-child,.metric td:first-child { text-align:left; }
  .metric th { color:#475569; position:sticky; top:0; background:#fff; }
  .table-scroll { max-height:390px; overflow:auto; }
  .detail { display:none; }
  .small { font-size:12px; color:var(--muted); }
  code { font-size:11px; word-break:break-all; }
  @media (max-width: 980px) { .layout { grid-template-columns:1fr; padding:12px; } .plot-wrap { height:60vh; } }
</style>
</head>
<body>
<header>
  <h1>9957/g0：ST 锚点 ↔ registered H&amp;E/UNI2 支撑区</h1>
  <p class="sub">可拖动平移；滚轮缩放；点击右侧图例可隐藏/显示切片。默认显示全部 ST 与各锚点对应的有效 H&amp;E/UNI2 特征网格边界。</p>
</header>
<main class="layout">
  <section>
    <div class="card">
      <h2>全局 XY 视图（高对比、不透明）</h2>
      <div class="controls">
        <button id="fitGlobal">恢复全局范围</button>
        <button id="focus5161">只看 51–61</button>
        <button id="showAll">显示全部</button>
        <button id="hideAll">隐藏全部</button>
        <label class="switch"><input id="showST" type="checkbox" checked> ST 点</label>
        <label class="switch"><input id="showHE" type="checkbox" checked> H&amp;E 支撑边界</label>
      </div>
      <div class="plot-wrap"><canvas id="globalPlot"></canvas></div>
      <div class="plot-caption"><span>X/Y 单位：µm；颜色按 section ID 区分</span><span id="globalStatus"></span></div>
    </div>
    <div class="card detail">
      <h2>51–61 局部放大视图</h2>
      <div class="controls"><button id="fitDetail">恢复 51–61 范围</button><label class="switch"><input id="detailST" type="checkbox" checked> ST 点</label><label class="switch"><input id="detailHE" type="checkbox" checked> H&amp;E 支撑边界</label></div>
      <div class="plot-wrap" style="height:min(62vh,650px)"><canvas id="detailPlot"></canvas></div>
      <div class="plot-caption"><span>红色：section 51；绿色：section 61</span><span id="detailStatus"></span></div>
    </div>
  </section>
  <aside>
    <div class="card">
      <h2>当前数据</h2>
      <div class="note">
        <b>注意：</b>黄色/彩色边界不是原始低清 mask 的直接变换，而是模型实际查询的 registered H&amp;E/UNI2 feature-grid 有效支撑区域。它和 ST 点使用同一个 repaired global coordinate frame。
      </div>
      <p class="small">坐标系：</p><code>__FRAME__</code>
      <p class="small" style="margin-top:10px">ST 细胞总数：<b>__TOTAL__</b><br>HTML 中显示 ST 点：<b>__SAMPLED__</b><br>H&amp;E 支撑边界点：<b>__SUPPORT__</b><br>加权 feature 有效率：<b>__COVERAGE__</b></p>
      <h2 style="margin-top:15px">切片显示</h2>
      <div id="legend" class="legend"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>切片级 QC</h2>
      <div class="table-scroll"><table class="metric"><thead><tr><th>section</th><th>cells</th><th>valid</th><th>q1</th><th>q2</th><th>q3</th></tr></thead><tbody id="metrics"></tbody></table></div>
      <p class="small" style="margin-top:10px">quality：q1=完整双线性查询，q2=部分有效角点，q3=有界最近邻恢复。</p>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>文件信息</h2>
      <p class="small">本 HTML 为离线单文件，可直接用浏览器打开，不需要 Plotly 或网络。</p>
      <p class="small">生成路径：</p><code>__OUTPUT__</code>
    </div>
  </aside>
</main>
<script>
const DATA = __DATA__;
const sections = DATA.sections;

function fmtPct(x) { return (100 * x).toFixed(2) + '%'; }
function rgba(hex, alpha) {
  const value = parseInt(hex.slice(1), 16);
  const r = (value >> 16) & 255, g = (value >> 8) & 255, b = value & 255;
  return `rgba(${r},${g},${b},${alpha})`;
}

function boundsFor(items) {
  let xs = [], ys = [];
  for (const s of items) {
    for (const p of s.st) { xs.push(p[0]); ys.push(p[1]); }
    for (const p of s.he) { xs.push(p[0]); ys.push(p[1]); }
  }
  if (!xs.length) return {xmin:0,xmax:1,ymin:0,ymax:1};
  let xmin=Math.min(...xs), xmax=Math.max(...xs), ymin=Math.min(...ys), ymax=Math.max(...ys);
  const dx=Math.max(xmax-xmin,1), dy=Math.max(ymax-ymin,1), pad=.06*Math.max(dx,dy);
  return {xmin:xmin-pad,xmax:xmax+pad,ymin:ymin-pad,ymax:ymax+pad};
}

class ScatterCanvas {
  constructor(canvas, items, statusNode) {
    this.canvas=canvas; this.ctx=canvas.getContext('2d'); this.items=items; this.statusNode=statusNode;
    this.visible=new Map(items.map(s=>[s.id,true])); this.showST=true; this.showHE=true;
    this.defaultBounds=boundsFor(items); this.view={...this.defaultBounds}; this.drag=null;
    this.resizeObserver = typeof ResizeObserver === 'function' ? new ResizeObserver(()=>this.resize()) : null;
    if (this.resizeObserver) this.resizeObserver.observe(canvas.parentElement);
    else window.addEventListener('resize', ()=>this.resize());
    this.bind(); this.resize();
  }
  bind() {
    this.canvas.addEventListener('pointerdown', e=>{this.canvas.setPointerCapture(e.pointerId); this.drag={x:e.offsetX,y:e.offsetY,view:{...this.view}}; this.canvas.classList.add('dragging');});
    this.canvas.addEventListener('pointerup', e=>{this.drag=null; this.canvas.classList.remove('dragging');});
    this.canvas.addEventListener('pointercancel', e=>{this.drag=null; this.canvas.classList.remove('dragging');});
    this.canvas.addEventListener('pointermove', e=>{
      if (!this.drag) return;
      const r=this.canvas.getBoundingClientRect(), sx=(this.view.xmax-this.view.xmin)/r.width, sy=(this.view.ymax-this.view.ymin)/r.height;
      const dx=(e.offsetX-this.drag.x)*sx, dy=(e.offsetY-this.drag.y)*sy;
      this.view.xmin=this.drag.view.xmin-dx; this.view.xmax=this.drag.view.xmax-dx; this.view.ymin=this.drag.view.ymin+dy; this.view.ymax=this.drag.view.ymax+dy; this.draw();
    });
    this.canvas.addEventListener('wheel', e=>{
      e.preventDefault(); const r=this.canvas.getBoundingClientRect();
      const px=e.offsetX/r.width, py=e.offsetY/r.height, factor=e.deltaY<0?.82:1.22;
      const x=this.view.xmin+px*(this.view.xmax-this.view.xmin), y=this.view.ymax-py*(this.view.ymax-this.view.ymin);
      const nw=(this.view.xmax-this.view.xmin)*factor, nh=(this.view.ymax-this.view.ymin)*factor;
      this.view.xmin=x-px*nw; this.view.xmax=x+(1-px)*nw; this.view.ymin=y-(1-py)*nh; this.view.ymax=y+py*nh; this.draw();
    }, {passive:false});
  }
  resize() { const r=this.canvas.getBoundingClientRect(), dpr=window.devicePixelRatio||1; this.canvas.width=Math.max(1,Math.floor(r.width*dpr)); this.canvas.height=Math.max(1,Math.floor(r.height*dpr)); this.ctx.setTransform(dpr,0,0,dpr,0,0); this.draw(); }
  reset(bounds=this.defaultBounds) { this.view={...bounds}; this.draw(); }
  setVisibility(id, value) { this.visible.set(Number(id),value); this.draw(); }
  map(p) { const r=this.canvas.getBoundingClientRect(); return [(p[0]-this.view.xmin)/(this.view.xmax-this.view.xmin)*r.width, r.height-(p[1]-this.view.ymin)/(this.view.ymax-this.view.ymin)*r.height]; }
  draw() {
    const r=this.canvas.getBoundingClientRect(), ctx=this.ctx; ctx.clearRect(0,0,r.width,r.height); ctx.fillStyle='#ffffff'; ctx.fillRect(0,0,r.width,r.height);
    const sx=r.width/(this.view.xmax-this.view.xmin), sy=r.height/(this.view.ymax-this.view.ymin), radius=Math.max(0.65,Math.min(3.0,1.0+0.0015*Math.min(sx,sy)));
    for (const s of this.items) {
      if (!this.visible.get(s.id)) continue;
      if (this.showHE) { ctx.fillStyle='#111111'; for (const p of s.he) { const q=this.map(p); if(q[0]>=-2&&q[0]<=r.width+2&&q[1]>=-2&&q[1]<=r.height+2) ctx.fillRect(q[0]-0.8,q[1]-0.8,1.6,1.6); } }
      if (this.showST) { ctx.fillStyle=s.color; for (const p of s.st) { const q=this.map(p); if(q[0]>=-2&&q[0]<=r.width+2&&q[1]>=-2&&q[1]<=r.height+2) { ctx.beginPath(); ctx.arc(q[0],q[1],radius,0,2*Math.PI); ctx.fill(); } } }
    }
    ctx.fillStyle='#111111'; ctx.font='12px system-ui'; ctx.fillText('X (µm)',r.width-62,r.height-10); ctx.save(); ctx.translate(12,54); ctx.rotate(-Math.PI/2); ctx.fillText('Y (µm)',0,0); ctx.restore();
    this.statusNode.textContent=`X ${this.view.xmin.toFixed(0)}–${this.view.xmax.toFixed(0)} µm · Y ${this.view.ymin.toFixed(0)}–${this.view.ymax.toFixed(0)} µm`;
  }
}

const global = new ScatterCanvas(document.getElementById('globalPlot'), sections, document.getElementById('globalStatus'));
const detailItems = sections.filter(s=>s.id===51 || s.id===61);
const detail = new ScatterCanvas(document.getElementById('detailPlot'), detailItems, document.getElementById('detailStatus'));

function setAllVisible(value) { for (const s of sections) { global.setVisibility(s.id,value); const box=document.getElementById('sec-'+s.id); if(box) box.checked=value; } }
document.getElementById('fitGlobal').onclick=()=>global.reset();
document.getElementById('showAll').onclick=()=>setAllVisible(true);
document.getElementById('hideAll').onclick=()=>setAllVisible(false);
document.getElementById('focus5161').onclick=()=>{ global.reset(boundsFor(detailItems)); setAllVisible(false); for(const id of [51,61]) { global.setVisibility(id,true); const box=document.getElementById('sec-'+id); if(box) box.checked=true; } };
document.getElementById('showST').onchange=e=>{global.showST=e.target.checked;global.draw();};
document.getElementById('showHE').onchange=e=>{global.showHE=e.target.checked;global.draw();};
document.getElementById('fitDetail').onclick=()=>detail.reset();
document.getElementById('detailST').onchange=e=>{detail.showST=e.target.checked;detail.draw();};
document.getElementById('detailHE').onchange=e=>{detail.showHE=e.target.checked;detail.draw();};

const legend=document.getElementById('legend');
for (const s of sections) {
  const label=document.createElement('label'); const box=document.createElement('input'); box.type='checkbox'; box.checked=true; box.id='sec-'+s.id; box.onchange=e=>{global.setVisibility(s.id,e.target.checked);};
  const sw=document.createElement('span'); sw.className='swatch'; sw.style.background=s.color; label.append(box,sw,document.createTextNode('section '+s.id)); legend.append(label);
}
const tbody=document.getElementById('metrics');
for (const s of sections) {
  const tr=document.createElement('tr'); tr.innerHTML=`<td>${s.id}</td><td>${s.n_cells.toLocaleString()}</td><td>${fmtPct(s.feature_valid_fraction)}</td><td>${fmtPct(s.quality_1_fraction)}</td><td>${fmtPct(s.quality_2_fraction)}</td><td>${fmtPct(s.quality_3_fraction)}</td>`; tbody.append(tr);
}
</script>
</body>
</html>'''.replace("__FRAME__", frame_label).replace("__TOTAL__", f"{payload['total_st_cells']:,}").replace("__SAMPLED__", f"{payload['sampled_st_points']:,}").replace("__SUPPORT__", f"{payload['support_boundary_points']:,}").replace("__COVERAGE__", coverage_label).replace("__OUTPUT__", generated_label).replace("__DATA__", data_json)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing HTML: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = _load_payload(args.candidate, args.feature_store)
    args.output.write_text(_build_html(payload, args.output), encoding="utf-8")
    print(json.dumps({
        "status": "created",
        "output": str(args.output.resolve()),
        "coordinate_frame": payload["coordinate_frame"],
        "total_st_cells": payload["total_st_cells"],
        "sampled_st_points": payload["sampled_st_points"],
        "support_boundary_points": payload["support_boundary_points"],
        "weighted_feature_valid_fraction": payload["weighted_feature_valid_fraction"],
    }, indent=2))


if __name__ == "__main__":
    main()
