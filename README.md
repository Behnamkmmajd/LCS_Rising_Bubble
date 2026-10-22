# Rising-bubble LCS

Code to compute Lagrangian coherent structures (LCS) in the wake of a 2D
axisymmetric rising bubble, from stored velocity snapshots:

- `FTLE2D.py` computes the finite-time Lyapunov exponent (FTLE) field and the
  repelling shrinklines that mark the wake boundary (hyperbolic LCS).
- `LAVD2D.py` computes the Lagrangian-averaged vorticity deviation (LAVD) field
  and the outermost closed contours around its maxima (elliptic LCS, i.e. the
  vortex boundaries).

The actual LCS machinery is the TBarrier library (the notebooks under
`TBarrier/`). This repository only adds the pieces specific to the rising-bubble
problem; those are collected in `rising_bubble_extensions.py`.

This folder is self-contained: the code, a sample case, and the output folder
are all here, so it can be copied anywhere and run as is.

## Folder layout

```
FTLE2D.py                      driver for the FTLE / hyperbolic LCS
LAVD2D.py                      driver for the LAVD / elliptic LCS
rising_bubble_extensions.py    helper routines added on top of TBarrier
TBarrier/                      the TBarrier notebooks (used unchanged)
database/
    3_2D_Ga60_Bo65_SD0_0_DR100_VR100/
        X.bin, Y.bin           grid
        Stacked_*.bin          velocity snapshots
        Phi/Stacked_*.bin      level-set snapshots
Run/                           results (.mat and .png) are written here
```

## What the extensions do

`rising_bubble_extensions.py` contains:

- reading of the binary grid, velocity and level-set files,
- the evaluation box, whose top follows the bubble surface,
- the seed box for the shrinklines, taken from the first FTLE peak just off the
  axis,
- the FTLE and LAVD field computations, split across the cores with joblib,
- a check that the in-plane stretching dominates the azimuthal stretching
  `(R1/R0)^2`, which is what justifies working in the meridional plane,
- writing the `.mat` and `.png` output.

The TBarrier routines (`gradient_flowmap`, `CauchyGreen`, `eigen`, `FTLE`,
`loc_max`, the tensorline integrator, `LAVD`, `find_outermost_contour`) are
imported and used without modification.

A note on the shrinkline seeding, since it uses two fields: the FTLE field only
locates the seed box, while the seeds themselves are the local maxima of the
larger Cauchy-Green eigenvalue inside that box. `min_distance` (TBarrier's own
parameter) then keeps the strongest one or two seeds.

## Requirements

Python with `numpy`, `scipy`, `matplotlib`, `numba`, `joblib`, `shapely`,
`scikit-image`, and `ipynb` (the last one is needed to import the TBarrier
notebooks as modules).

```
conda create -n tbarrier python=3.9 numpy scipy matplotlib numba joblib shapely scikit-image -y
conda activate tbarrier
pip install ipynb
```

## Running

```
python FTLE2D.py
python LAVD2D.py
```

Each driver starts with a block of input variables (case name, snapshot range,
grid spacing, number of cores, and so on). Results go to `Run/FTLE2D/<case>/`
and `Run/LAVD2D/<case>/`.

## Sample data

The included case `3_2D_Ga60_Bo65_SD0_0_DR100_VR100` only ships with snapshots
60 to 578, which is enough for the default FTLE run (178 to 578) and LAVD run
(60 to 330). To use snapshots outside that range, copy the matching
`Stacked_<i>.bin` files into `database/<case>/` and `database/<case>/Phi/`.

## References

- G. Haller, Lagrangian coherent structures, Annu. Rev. Fluid Mech. (2015).
- M. Farazmand and G. Haller, Computing Lagrangian coherent structures from
  their variational theory, Chaos (2012).
- G. Haller et al., Defining coherent vortices objectively from the vorticity,
  J. Fluid Mech. (2016).
