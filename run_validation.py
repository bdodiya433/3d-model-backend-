"""
run_validation.py — Standalone script to:
  1. Run the full reconstruction pipeline with new photo-accurate texturing
  2. Save before/after textures for comparison
  3. Compute SSIM/PSNR between reference photos and rendered views
  4. Generate side-by-side comparison images

Usage:
  python run_validation.py \
    --input  temp/1d54cc73-22cd-458a-a915-66bd5e51b93a_rec_in \
    --output temp/validation_new \
    --compare temp/1d54cc73-22cd-458a-a915-66bd5e51b93a_rec_out
"""

import argparse
import json
import sys
import shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.resolve()
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


# ─── SSIM / PSNR helpers ──────────────────────────────────────────────────────

def _to_float(arr):
    return arr.astype(np.float64) / 255.0


def compute_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Structural Similarity Index (luminance channel only, simplified)."""
    a = _to_float(img_a.mean(axis=2) if img_a.ndim == 3 else img_a)
    b = _to_float(img_b.mean(axis=2) if img_b.ndim == 3 else img_b)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = a.mean(), b.mean()
    sig_a = a.std() ** 2
    sig_b = b.std() ** 2
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    ssim = ((2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)) / \
           ((mu_a ** 2 + mu_b ** 2 + c1) * (sig_a + sig_b + c2))
    return float(ssim)


def compute_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((_to_float(img_a) - _to_float(img_b)) ** 2)
    if mse < 1e-12:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def make_side_by_side(img_a: np.ndarray, img_b: np.ndarray, label_a: str, label_b: str) -> np.ndarray:
    """Create a side-by-side comparison image with labels."""
    from PIL import Image, ImageDraw, ImageFont
    h = max(img_a.shape[0], img_b.shape[0])
    # Resize both to same height
    pil_a = Image.fromarray(img_a).resize((int(img_a.shape[1] * h / img_a.shape[0]), h), Image.Resampling.LANCZOS)
    pil_b = Image.fromarray(img_b).resize((int(img_b.shape[1] * h / img_b.shape[0]), h), Image.Resampling.LANCZOS)
    bar_h = 36
    combined_w = pil_a.width + pil_b.width + 4  # 4px gap
    out = Image.new("RGB", (combined_w, h + bar_h), (30, 30, 30))
    out.paste(pil_a, (0, bar_h))
    out.paste(pil_b, (pil_a.width + 4, bar_h))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, pil_a.width + 1, bar_h], fill=(50, 50, 80))
    draw.rectangle([pil_a.width + 3, 0, combined_w, bar_h], fill=(50, 80, 50))
    draw.text((8, 8), label_a, fill=(200, 200, 255))
    draw.text((pil_a.width + 12, 8), label_b, fill=(200, 255, 200))
    return np.array(out)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True,  help="Directory with front/back/left/right/top/bottom.png")
    parser.add_argument("--output",  required=True,  help="New output directory for photo-accurate pipeline")
    parser.add_argument("--compare", default=None,   help="Previous output directory for before/after comparison")
    parser.add_argument("--res",     type=int, default=4096, help="Texture resolution")
    args = parser.parse_args()

    input_dir  = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    compare_dir = Path(args.compare).resolve() if args.compare else None

    output_dir.mkdir(parents=True, exist_ok=True)
    compare_out = output_dir / "compare"
    compare_out.mkdir(parents=True, exist_ok=True)

    # ── Save BEFORE textures if compare_dir supplied ──────────────────────
    if compare_dir:
        old_tex = compare_dir / "texture" / "texture_basecolor.png"
        old_renders_dir = compare_dir / "renders"
        if old_tex.exists():
            shutil.copy2(old_tex, compare_out / "old_basecolor.png")
            print(f"[COMPARE] Saved old BaseColor → {compare_out}/old_basecolor.png")

    # ── Run the reconstruction pipeline ───────────────────────────────────
    from src.main import run_reconstruction
    print(f"\n{'='*70}")
    print(f"RUNNING PHOTO-ACCURATE PIPELINE")
    print(f"  Input:      {input_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Resolution: {args.res}×{args.res}")
    print(f"{'='*70}\n")

    result = run_reconstruction(
        input_dir=input_dir,
        output_dir=output_dir,
        texture_resolution=args.res,
    )

    # ── Copy NEW textures into compare dir ────────────────────────────────
    new_tex = result["texture"]["basecolor"]
    if Path(new_tex).exists():
        shutil.copy2(new_tex, compare_out / "new_basecolor.png")
        print(f"[COMPARE] Saved new BaseColor → {compare_out}/new_basecolor.png")

    # ── Copy renders into compare dir ─────────────────────────────────────
    view_names = ["front", "back", "left", "right", "top", "bottom"]
    renders = result.get("renders", {})
    for vname in view_names:
        rpath = renders.get(vname)
        if rpath and Path(rpath).exists():
            shutil.copy2(rpath, compare_out / f"new_{vname}.png")

    if compare_dir:
        for vname in view_names:
            old_r = compare_dir / "renders" / f"{vname}.png"
            if old_r.exists():
                shutil.copy2(old_r, compare_out / f"old_{vname}.png")

    # ── Generate side-by-side comparisons & compute metrics ───────────────
    from PIL import Image

    print(f"\n{'='*70}")
    print("VISUAL VALIDATION METRICS")
    print(f"{'='*70}")

    report = {}

    # BaseColor texture comparison
    old_bc_p = compare_out / "old_basecolor.png"
    new_bc_p = compare_out / "new_basecolor.png"
    if old_bc_p.exists() and new_bc_p.exists():
        old_bc = np.array(Image.open(old_bc_p).convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS))
        new_bc = np.array(Image.open(new_bc_p).convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS))
        # Color distribution stats
        old_mean = old_bc.mean(axis=(0, 1))
        new_mean = new_bc.mean(axis=(0, 1))
        ssim_bc  = compute_ssim(old_bc, new_bc)
        print(f"\n[BaseColor Texture]")
        print(f"  OLD mean RGB : ({old_mean[0]:.1f}, {old_mean[1]:.1f}, {old_mean[2]:.1f})")
        print(f"  NEW mean RGB : ({new_mean[0]:.1f}, {new_mean[1]:.1f}, {new_mean[2]:.1f})")
        print(f"  SSIM         : {ssim_bc:.4f}  (1.0=identical)")

        # Artificial-red check: count pixels very close to flat #F40000
        r, g, b = new_bc[..., 0], new_bc[..., 1], new_bc[..., 2]
        flat_red_mask = (r > 220) & (g < 30) & (b < 30)
        flat_red_pct  = float(flat_red_mask.sum()) / (1024 * 1024) * 100
        print(f"  Flat-red pixels (>220,<30,<30): {flat_red_pct:.2f}%  (should be ~0% for photo-accurate)")

        # Pure-white check: forced #FFFFFF pixels
        pure_white_mask = (r > 250) & (g > 250) & (b > 250)
        pure_white_pct  = float(pure_white_mask.sum()) / (1024 * 1024) * 100
        print(f"  Pure-white pixels (>250 all):   {pure_white_pct:.2f}%  (should be low for photo-accurate)")

        # Side-by-side
        sbs = make_side_by_side(old_bc, new_bc, "BEFORE (synthetic)", "AFTER (photo-accurate)")
        Image.fromarray(sbs).save(compare_out / "compare_basecolor.png")
        print(f"  Side-by-side → {compare_out}/compare_basecolor.png")

        report["basecolor"] = {
            "old_mean_rgb": old_mean.tolist(),
            "new_mean_rgb": new_mean.tolist(),
            "ssim": ssim_bc,
            "flat_red_pct": flat_red_pct,
            "pure_white_pct": pure_white_pct,
        }

    # Per-view render comparisons
    print(f"\n[Rendered View Comparisons vs. Reference Photos]")
    view_metrics = {}
    for vname in view_names:
        ref_p    = input_dir / f"{vname}.png"
        new_r_p  = compare_out / f"new_{vname}.png"
        old_r_p  = compare_out / f"old_{vname}.png"

        if not new_r_p.exists():
            print(f"  {vname:<8}: no new render found")
            continue

        new_r = np.array(Image.open(new_r_p).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS))

        if ref_p.exists():
            ref = np.array(Image.open(ref_p).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS))
            ssim_v = compute_ssim(ref, new_r)
            psnr_v = compute_psnr(ref, new_r)
            print(f"  {vname:<8}: SSIM={ssim_v:.4f}  PSNR={psnr_v:.2f}dB  (vs reference photo)")
            view_metrics[vname] = {"ssim_vs_ref": ssim_v, "psnr_vs_ref": psnr_v}

            sbs = make_side_by_side(ref, new_r, f"Reference Photo ({vname})", f"New Render ({vname})")
            Image.fromarray(sbs).save(compare_out / f"compare_{vname}_vs_ref.png")

        if old_r_p.exists():
            old_r = np.array(Image.open(old_r_p).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS))
            ssim_old = compute_ssim(old_r, new_r)
            print(f"  {vname:<8}: SSIM={ssim_old:.4f}  (old vs new render)")
            sbs2 = make_side_by_side(old_r, new_r, f"Old Render ({vname})", f"New Render ({vname})")
            Image.fromarray(sbs2).save(compare_out / f"compare_{vname}_old_vs_new.png")

    report["views"] = view_metrics

    # ── Save JSON report ───────────────────────────────────────────────────
    report_path = compare_out / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print(f"VALIDATION COMPLETE")
    print(f"  Compare dir : {compare_out}")
    print(f"  GLB         : {result['final']['glb']}")
    print(f"  Report      : {report_path}")
    print(f"  Elapsed     : {result['elapsed_seconds']:.1f}s")
    print(f"{'='*70}\n")

    return result


if __name__ == "__main__":
    main()
