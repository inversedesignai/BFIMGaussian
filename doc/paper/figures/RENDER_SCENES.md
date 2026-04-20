# Rendering the three case-study scene figures

**Audience:** a Claude session (or human) on a workstation with Julia and a working GPU. Your job is to render the three 3D scenes and commit the PNGs so the paper can rebuild.

**Context:** `paper.tex` now references three figures that do not yet exist on disk:

- `doc/paper/figures/radar_scene.png` (Case Study A)
- `doc/paper/figures/scqubit_scene.png` (Case Study B)
- `doc/paper/figures/photonic_3d_scene.png` (Case Study C)

Until these PNGs are produced, `paper.tex` will **not compile** — `\includegraphics` on a missing file is a fatal LaTeX error. The three producing scripts are already committed alongside this document:

- [`make_radar_figure.jl`](make_radar_figure.jl)
- [`make_scqubit_figure.jl`](make_scqubit_figure.jl)
- [`make_photonic_3d_figure.jl`](make_photonic_3d_figure.jl)

## Steps

### 1. Pull the latest repo

```bash
cd ~/BFIMGaussian
git pull origin master
```

### 2. Install the Makie stack

From any writable Julia project (or `cd` into `doc/paper/figures` and instantiate there), ensure the following packages are available:

```julia
using Pkg
Pkg.add(["GLMakie", "Serialization", "Statistics", "LinearAlgebra"])
```

GLMakie requires OpenGL. If the workstation is headless / has no X server, either:
- **Preferred**: run the scripts through a minimal X server (`xvfb-run julia script.jl`), or
- **Fallback**: replace `using GLMakie; GLMakie.activate!()` with `using CairoMakie; CairoMakie.activate!()` at the top of each script. CairoMakie's 3D rendering is less polished but works without a display. Do not commit that change; it is only a local workaround.

### 3. Render the three PNGs

From the repo root:

```bash
julia --project=. doc/paper/figures/make_radar_figure.jl
julia --project=. doc/paper/figures/make_scqubit_figure.jl
julia --project=. doc/paper/figures/make_photonic_3d_figure.jl
```

Each script runs in under a minute on a single GPU. Each prints `Saved: <path>` on success.

**For the photonic script**, the `BFIM_REPO` environment variable defaults to two-directories-up from the script (which should resolve to `~/BFIMGaussian`). If the autodetect fails, set it explicitly:

```bash
BFIM_REPO=$HOME/BFIMGaussian julia --project=. doc/paper/figures/make_photonic_3d_figure.jl
```

The script will try to load the β=256 autotune checkpoint from `checkpoints/eps_geom_step_00580.jls`. If that file is missing, the script falls back to a synthetic pattern (and prints a warning).

### 4. Verify the three PNGs exist

```bash
ls -lh doc/paper/figures/radar_scene.png \
       doc/paper/figures/scqubit_scene.png \
       doc/paper/figures/photonic_3d_scene.png
```

Each should be several hundred KB. Eyeball them by opening each in an image viewer. Check specifically:

- **radar_scene.png**: 16 gold elements on a grey baseplate, ring of 16 spheres (one red, one blue), two translucent beam lobes emerging from the array center.
- **scqubit_scene.png**: blue substrate, orange cross electrode, SQUID loop with two red dots, purple bias line, navy field arc, green meander, and a Bloch-sphere inset on the right.
- **photonic_3d_scene.png**: a slab of dark silicon pillars on a light cladding, four waveguide stubs on the four sides, red arrows on two of them, a translucent plasma-colormap overlay above.

If any scene looks obviously wrong (empty, truncated, single color) — the most likely cause is GLMakie failing to initialize OpenGL on a headless node. Try the `CairoMakie` fallback above, or run on a node with a display.

### 5. Build the paper

```bash
cd doc/paper
cp paper.tex tmp_build.tex
cp paper.bib tmp_build.bib
pdflatex -interaction=nonstopmode -halt-on-error tmp_build.tex
bibtex tmp_build
pdflatex -interaction=nonstopmode -halt-on-error tmp_build.tex
pdflatex -interaction=nonstopmode -halt-on-error tmp_build.tex
mv tmp_build.pdf paper.pdf
rm -f tmp_build.* *.aux *.log *.out *.toc *.bbl *.blg
```

(The `tmp_build` detour avoids a Google Drive sync lock on Windows; on Linux, `pdflatex paper.tex` directly is fine.)

Expected: **29 pages** PDF, no fatal errors, three new figures inserted at the top of each Case Study section.

### 6. Commit and push

```bash
git add doc/paper/figures/radar_scene.png \
        doc/paper/figures/scqubit_scene.png \
        doc/paper/figures/photonic_3d_scene.png \
        doc/paper/paper.pdf
git commit -m "paper/figures: render three case-study scene PNGs"
git push origin master
```

## What to report back

Once done, report (short, 3–4 lines):

1. PNG file sizes: "radar_scene.png = XXX KB; scqubit_scene.png = YYY KB; photonic_3d_scene.png = ZZZ KB."
2. Paper built cleanly: "paper.pdf rebuilt at N pages, no undefined references beyond the pre-existing `bhargava2024topology`."
3. Any deviations: CairoMakie fallback used, script edits made, or render quality concerns.

## Troubleshooting

**"ERROR: LoadError: UndefVarError: `voxels!` not defined"**: your Makie version is older than v0.20. Upgrade with `Pkg.update("GLMakie")` and retry. The `voxels!` primitive was added in Makie 0.20.

**"ERROR: MethodError: no method matching ... for FastShading"**: this is a Makie API-rename issue between versions. Replace `FastShading` with `MultiLightShading` or `NoShading`, whichever your version accepts. This is cosmetic only.

**"cannot open display: :0"**: headless node without X. Either `xvfb-run` the script or switch to CairoMakie (see step 2 fallback).

**Photonic script loads `synthetic fallback`**: the β=256 checkpoint is not at the expected path. Either (a) verify `ls ~/BFIMGaussian/checkpoints/eps_geom_step_00580.jls` and set `BFIM_REPO` if needed, or (b) accept the synthetic pattern — it is visually similar to the real optimized design and acceptable for the figure.

**Paper builds but figures look oversized / tiny**: the `\includegraphics[width=0.XX\textwidth]{}` argument in `paper.tex` can be tuned. Values currently set: radar 0.85, scqubit 0.92, photonic 0.92. Halve each if they overflow.

---

Rendered on: (fill in after running)
