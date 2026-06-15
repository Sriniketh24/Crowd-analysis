"""Build a fully self-contained Colab T4 handoff notebook for the v7 head-ID pipeline.

Usage:
    python scripts/build_handoff_notebook.py

The script:
1. Reads the three pipeline source files from the working tree.
2. Smoke-tests them by writing to a temp dir and running the script with --max-frames 8.
3. If (and only if) the smoke test passes, writes colab_handoff_t4.ipynb at repo root.
"""

import datetime
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]

FILES_TO_EMBED = [
    "src/vision/__init__.py",
    "src/vision/detector.py",
    "scripts/run_head_id_stability.py",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def code_cell(source_lines: list[str]) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }


def md_cell(source_lines: list[str]) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines,
    }


def writefile_cell(rel_path: str, content: str) -> dict:
    """Produce a %%writefile magic cell that embeds file content verbatim."""
    header = f"%%writefile {rel_path}\n"
    body_lines = content.splitlines(keepends=True)
    return code_cell([header] + body_lines)


# ---------------------------------------------------------------------------
# STEP 1 — read sources
# ---------------------------------------------------------------------------

print("Reading source files …")
embedded: dict[str, str] = {}
for rel in FILES_TO_EMBED:
    path = REPO / rel
    if not path.exists():
        print(f"ERROR: source file not found: {path}")
        sys.exit(1)
    embedded[rel] = path.read_text(encoding="utf-8")
    print(f"  {rel}: {len(embedded[rel].splitlines())} lines")


# ---------------------------------------------------------------------------
# STEP 2 — smoke test
# ---------------------------------------------------------------------------

print("\nRunning smoke test …")
TMP = pathlib.Path(tempfile.mkdtemp(prefix="handoff_smoke_"))
try:
    # Write embedded files into temp dir
    for rel, content in embedded.items():
        dest = TMP / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

    venv_python = REPO / ".venv" / "bin" / "python"
    smoke_python = str(venv_python) if venv_python.exists() else sys.executable

    smoke_cmd = [
        smoke_python,
        str(TMP / "scripts" / "run_head_id_stability.py"),
        "--video", str(REPO / "data" / "input_videos" / "sample.mp4"),
        "--model", str(REPO / "models" / "fine_tuned" / "head_detector_s" / "weights" / "best.pt"),
        "--arms", "baseline", "fix_0p16_wide_guard_app",
        "--device", "cpu",
        "--imgsz", "640",
        "--max-frames", "8",
        "--no-draw-annotated",
        "--no-make-side-by-side",
        "--output-root", str(TMP / "smoke_out"),
    ]

    print(f"  python: {smoke_python}")
    print(f"  cwd:    {TMP}")
    print(f"  cmd:    {' '.join(smoke_cmd[1:3])} …")

    result = subprocess.run(
        smoke_cmd,
        cwd=str(TMP),
        capture_output=True,
        text=True,
        timeout=600,
    )

    # The script derives the model label from the path stem when it is an absolute
    # path that does not match DEFAULT_RPEE_WEIGHTS (which is relative).  In the
    # smoke run we pass the absolute repo path, so the label is "best".
    expected_csv_rpee = TMP / "smoke_out" / "sample" / "rpee_head_s" / "comparison_metrics.csv"
    expected_csv_best = TMP / "smoke_out" / "sample" / "best" / "comparison_metrics.csv"
    expected_csv = expected_csv_rpee if expected_csv_rpee.exists() else expected_csv_best
    smoke_ok = (result.returncode == 0) and expected_csv.exists()

    if not smoke_ok:
        combined = result.stdout + result.stderr
        tail = combined.splitlines()[-30:]
        print("\n=== SMOKE TEST FAILED — last 30 lines of output ===")
        for line in tail:
            print(line)
        print("====================================================")
        print(f"returncode: {result.returncode}")
        print(f"expected CSV exists: {expected_csv.exists()}")
        sys.exit(1)

    print("  SMOKE TEST PASSED")
finally:
    shutil.rmtree(TMP, ignore_errors=True)


# ---------------------------------------------------------------------------
# STEP 3 — provenance
# ---------------------------------------------------------------------------

commit_result = subprocess.run(
    ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True,
)
commit = commit_result.stdout.strip() or "unknown"

dirty_result = subprocess.run(
    ["git", "-C", str(REPO), "status", "--porcelain"],
    capture_output=True, text=True,
)
dirty = bool(dirty_result.stdout.strip())

build_utc = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
dirty_suffix = " (+uncommitted changes)" if dirty else ""

print(f"\nProvenance: commit={commit}{dirty_suffix}  built={build_utc}")


# ---------------------------------------------------------------------------
# STEP 4 — build notebook cells
# ---------------------------------------------------------------------------

cells: list[dict] = []

# Cell 1 — title / intro markdown
cells.append(md_cell([
    "# Crowd Head-ID Tracking — Self-Contained Colab T4 Handoff (v7)\n",
    "\n",
    "This notebook is **fully self-contained** — there is **NO git clone**.\n",
    "The pipeline source (`src/vision/__init__.py`, `src/vision/detector.py`,\n",
    "`scripts/run_head_id_stability.py`) is written into this Colab session\n",
    "by the `%%writefile` cells below, then executed directly.\n",
    "\n",
    "## Quick start\n",
    "1. **Runtime → Change runtime type → T4 GPU**\n",
    "2. **Runtime → Run all** (≈ 5–10 min for the full video, ~1 min for --max-frames 200)\n",
    "\n",
    "## Reference numbers\n",
    "| arm | confirmed_unique | inflation_factor |\n",
    "|-----|-----------------|------------------|\n",
    "| baseline | ~338 | ~3.1× |\n",
    "| **fix_0p16_wide_guard_app (v7 winner)** | **~125** | **~3.1×** |\n",
    "| A100 best (same model/config) | **105** | **2.5×** |\n",
    "\n",
    f"Generated by `scripts/build_handoff_notebook.py` from commit "
    f"`{commit}`{dirty_suffix} at {build_utc}.\n",
]))

# Cell 2 — GPU check
cells.append(code_cell([
    "import torch\n",
    "!nvidia-smi -L\n",
    'assert torch.cuda.is_available(), "No GPU! Runtime > Change runtime type > T4 GPU, then Run all."\n',
    'print("GPU:", torch.cuda.get_device_name(0))\n',
]))

# Cell 3 — section header
cells.append(md_cell([
    "## Step 1 — write the pipeline source into this Colab session (no repo clone)\n",
]))

# Cell 4 — create dirs
cells.append(code_cell([
    "import os\n",
    'os.makedirs("src/vision", exist_ok=True)\n',
    'os.makedirs("scripts", exist_ok=True)\n',
    'print("dirs ready")\n',
]))

# Cell 5 — %%writefile src/vision/__init__.py
cells.append(writefile_cell("src/vision/__init__.py", embedded["src/vision/__init__.py"]))

# Cell 6 — %%writefile src/vision/detector.py
cells.append(writefile_cell("src/vision/detector.py", embedded["src/vision/detector.py"]))

# Cell 7 — %%writefile scripts/run_head_id_stability.py
cells.append(writefile_cell("scripts/run_head_id_stability.py", embedded["scripts/run_head_id_stability.py"]))

# Cell 8 — pip install
cells.append(code_cell([
    "# Pin versions; can be relaxed to ultralytics>=8.4 supervision>=0.28 if pip complains\n",
    '%pip install -q "ultralytics==8.4.58" "supervision==0.28.0" "tqdm>=4.66"\n',
    "import ultralytics, supervision\n",
    'print("ultralytics", ultralytics.__version__, "| supervision", supervision.__version__)\n',
]))

# Cell 9 — section header
cells.append(md_cell([
    "## Step 2 — get the model + sample video (from the v7-handoff release)\n",
]))

# Cell 10 — download assets
cells.append(code_cell([
    "import os, urllib.request, pathlib\n",
    "\n",
    'BASE = "https://github.com/Sriniketh24/Crowd-analysis/releases/download/v7-handoff"\n',
    "assets = {\n",
    '    "models/fine_tuned/head_detector_s/weights/best.pt": f"{BASE}/head_detector_s_best.pt",\n',
    '    "data/input_videos/sample.mp4": f"{BASE}/sample.mp4",\n',
    "}\n",
    "for dest_rel, url in assets.items():\n",
    "    dest = pathlib.Path(dest_rel)\n",
    "    dest.parent.mkdir(parents=True, exist_ok=True)\n",
    "    if dest.exists() and dest.stat().st_size > 0:\n",
    '        print(f"  already present: {dest_rel} ({dest.stat().st_size / 1e6:.1f} MB)")\n',
    "        continue\n",
    '    print(f"  downloading {dest_rel} …")\n',
    "    urllib.request.urlretrieve(url, dest)\n",
    '    print(f"  → {dest.stat().st_size / 1e6:.1f} MB")\n',
]))

# Cell 11 — section header
cells.append(md_cell([
    "## Step 3 — run the pipeline on the GPU\n",
]))

# Cell 12 — run command (single line, no backslash continuation)
cells.append(code_cell([
    "# To run only the winner: --arms fix_0p16_wide_guard_app\n",
    "!python scripts/run_head_id_stability.py --video data/input_videos/sample.mp4 --model models/fine_tuned/head_detector_s/weights/best.pt --arms baseline fix_0p16_wide_guard_app --device cuda --half --imgsz 1536 --output-root data/outputs/handoff_run\n",
]))

# Cell 13 — metrics table
cells.append(code_cell([
    "import pandas as pd\n",
    'm = pd.read_csv("data/outputs/handoff_run/sample/rpee_head_s/comparison_metrics.csv")\n',
    'cols = ["arm", "confirmed_unique", "inflation_factor", "peak_concurrent_confirmed",\n',
    '        "residual_switch_events", "duplicate_id_frames", "gt_mae"]\n',
    "m[cols]\n",
]))

# Cell 14 — download annotated video
cells.append(code_cell([
    "from google.colab import files\n",
    'OUT = "data/outputs/handoff_run/sample/rpee_head_s"\n',
    'v7 = f"{OUT}/fix_0p16_wide_guard_app_annotated.mp4"\n',
    'print("v7 annotated video:", v7)\n',
    "files.download(v7)\n",
    "# also: files.download(f\"{OUT}/side_by_side.mp4\")  for the comparison\n",
]))

# Cell 15 — Drive section header
cells.append(md_cell([
    "## Optional: persist outputs to Google Drive\n",
    "\n",
    "Mount Google Drive and copy the run outputs so they survive when the runtime recycles.\n",
]))

# Cell 16 — Drive mount + copy
cells.append(code_cell([
    "from google.drive import drive  # type: ignore\n",
    "drive.mount('/content/drive')\n",
    "import shutil, pathlib\n",
    "src = pathlib.Path('data/outputs/handoff_run')\n",
    "dst = pathlib.Path('/content/drive/MyDrive/crowd-analysis-outputs/handoff_run')\n",
    "shutil.copytree(str(src), str(dst), dirs_exist_ok=True)\n",
    "print('Saved to:', dst)\n",
]))

# Cell 17 — notes & troubleshooting
cells.append(md_cell([
    "## Notes & troubleshooting\n",
    "\n",
    "- **All arms** are defined in the embedded `scripts/run_head_id_stability.py` under `DEFAULT_ARMS`.\n",
    "  The v7 winner is `fix_0p16_wide_guard_app`.\n",
    "- **Your own video**: change `--video` to any `.mp4` path; note that `gt_mae`/`MANUAL_GT`\n",
    "  is calibrated to `sample.mp4` only — ignore it for other videos.\n",
    "- **Speed tips**: use `--imgsz 1280` or add `--max-frames 200` for a quick preview.\n",
    "- **Pip pin fallback**: if pip refuses the exact pins, relax to\n",
    "  `ultralytics>=8.4 supervision>=0.28`.\n",
    f"- This notebook is generated — to update it, edit the pipeline in the repo and re-run\n",
    f"  `python scripts/build_handoff_notebook.py` (commit `{commit}`).\n",
]))


# ---------------------------------------------------------------------------
# STEP 5 — write notebook
# ---------------------------------------------------------------------------

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

out_path = REPO / "colab_handoff_t4.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Verify it's valid JSON
with open(out_path, encoding="utf-8") as f:
    json.load(f)
print("notebook JSON: OK")

# ---------------------------------------------------------------------------
# STEP 6 — summary
# ---------------------------------------------------------------------------

nb_size_kb = out_path.stat().st_size / 1024
print("\n=== BUILD COMPLETE ===")
print(f"  smoke:    PASS")
print(f"  commit:   {commit}{dirty_suffix}")
print(f"  built:    {build_utc}")
print(f"  cells:    {len(cells)}")
for rel, content in embedded.items():
    print(f"  {rel}: {len(content.splitlines())} lines embedded")
print(f"  output:   {out_path}  ({nb_size_kb:.1f} KB)")
