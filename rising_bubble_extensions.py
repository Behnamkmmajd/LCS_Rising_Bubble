# Helper routines for the rising-bubble LCS analysis.
#
# These build on the TBarrier notebooks: data loading, the automatic FTLE
# domain and seed box, parallel FTLE/LAVD field computations, the axisymmetric
# azimuthal-stretch check, and writing the results. The TBarrier functions
# imported below are used as they are.

import os
import sys
import numpy as np


def setup_tbarrier_paths(base_dir=None):
    # Put the TBarrier notebook folders on sys.path so "ipynb.fs.defs.*" imports work.
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    tb2d = os.path.join(base_dir, 'TBarrier', '2D')
    for sub in [('subfunctions', 'utils'),
                ('subfunctions', 'integration'),
                ('demos', 'AdvectiveBarriers', 'FTLE2D'),
                ('demos', 'AdvectiveBarriers', 'HyperbolicLCS'),
                ('demos', 'AdvectiveBarriers', 'LAVD2D')]:
        p = os.path.join(tb2d, *sub)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


# Do this at import time too, because joblib starts fresh worker processes that
# re-import this module and need the TBarrier imports below to resolve.
setup_tbarrier_paths()

from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator

from ipynb.fs.defs.Interpolant import interpolant_unsteady
from ipynb.fs.defs.gradient_flowmap import gradient_flowmap
from ipynb.fs.defs.integration_dFdt import integration_dFdt
from ipynb.fs.defs.gradient_velocity import gradient_velocity
from ipynb.fs.defs.CauchyGreen import CauchyGreen
from ipynb.fs.defs.eigen import eigen
from ipynb.fs.defs.FTLE import _FTLE_C
from ipynb.fs.defs.LAVD import _LAVD
from ipynb.fs.defs.orient_vectorfield import _orient_vectorfield
from ipynb.fs.defs.RK4_tensorlines import _RK4_tensorlines_incompressible


def split_evenly(a, n):
    # Split array "a" into n roughly equal chunks (used to hand work to the cores).
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


# -------------------------------------------------------------------------
# Reading the binary database
# -------------------------------------------------------------------------
def resolve_database_folder(folder_name):
    # The case folders live in database/ next to this file.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'database', folder_name)


def lcs_rising_bubble_root():
    # Root of the bundle; Run/ outputs go here.
    return os.path.dirname(os.path.abspath(__file__))


def load_grid(read_folder):
    XX = np.fromfile(os.path.join(read_folder, 'X.bin'), dtype=np.float64)
    YY = np.fromfile(os.path.join(read_folder, 'Y.bin'), dtype=np.float64)
    return XX, YY


def load_velocity_snapshots(read_folder, snapshot_file_offset, start_idx, end_idx, nx, ny, flip_snapshot):
    # Read the velocity snapshots between the two indices into U, V of shape (ny, nx, nt).
    lower = min(start_idx, end_idx)
    upper = max(start_idx, end_idx)
    nt = upper - lower + 1
    U = np.zeros((ny, nx, nt))
    V = np.zeros((ny, nx, nt))
    npts = nx * ny
    order = 'F' if flip_snapshot else 'C'

    for i in range(lower, upper + 1):
        fname = os.path.join(read_folder, f"Stacked_{i + snapshot_file_offset}.bin")
        raw = np.fromfile(fname, dtype=np.float64)
        t = i - lower
        U[:, :, t] = raw[:npts].reshape((ny, nx), order=order)
        V[:, :, t] = raw[npts:].reshape((ny, nx), order=order)
    return U, V


def load_phi_snapshots(read_folder, snapshot_file_offset, start_idx, end_idx, nx, ny, flip_snapshot):
    # Read the level-set field (phi > 0 is inside the bubble). We only use it to
    # place the top of the FTLE domain at the bubble surface.
    lower = min(start_idx, end_idx)
    upper = max(start_idx, end_idx)
    nt = upper - lower + 1
    PHI = np.zeros((ny, nx, nt))

    phi_folder = None
    for name in ('PHI', 'Phi'):
        candidate = os.path.join(read_folder, name)
        if os.path.isdir(candidate):
            phi_folder = candidate
            break
    if phi_folder is None:
        raise FileNotFoundError(f"No PHI/Phi folder inside {read_folder}")

    npts = nx * ny
    order = 'F' if flip_snapshot else 'C'
    for i in range(lower, upper + 1):
        fname = os.path.join(phi_folder, f"Stacked_{i + snapshot_file_offset}.bin")
        raw = np.fromfile(fname, dtype=np.float64)
        t = i - lower
        PHI[:, :, t] = raw[:npts].reshape((ny, nx), order=order)
    return PHI


# -------------------------------------------------------------------------
# FTLE/LAVD evaluation domain from the bubble position
# -------------------------------------------------------------------------
def compute_bubble_top_y(PHI_snap, Y):
    # Highest y that is still inside the bubble.
    inside = PHI_snap > 0
    if not np.any(inside):
        raise ValueError("This phi snapshot has no points inside the bubble")
    return float(np.max(Y[inside]))


def derive_ftle_y_bounds_from_bubble(PHI_snap, Y, top_padding, domain_height):
    # Put the top of the box a little above the bubble and go down by domain_height.
    bubble_top_y = compute_bubble_top_y(PHI_snap, Y)
    ymax = bubble_top_y + top_padding
    ymin = ymax - domain_height

    if ymin < float(np.min(Y)) or ymax > float(np.max(Y)):
        raise ValueError(f"FTLE y-bounds [{ymin:.4f}, {ymax:.4f}] are outside the data")
    return bubble_top_y, float(ymin), float(ymax)


def build_ftle_domain(xmin, xmax, ymin, ymax, dx_eval, dy_eval, X, Y, t0, tN, time_data):
    # Uniform grid of tracer start points. A few sanity checks first.
    assert np.min(X) <= xmin <= np.max(X) and np.min(X) <= xmax <= np.max(X), "x-bounds outside data"
    assert np.min(Y) <= ymin <= np.max(Y) and np.min(Y) <= ymax <= np.max(Y), "y-bounds outside data"
    assert np.min(time_data) <= min(t0, tN) and max(t0, tN) <= np.max(time_data), "time window outside data"
    assert dx_eval > 0 and dy_eval > 0, "grid spacing must be positive"

    Nx = max(2, int(np.floor((xmax - xmin) / dx_eval)) + 1)
    Ny = max(2, int(np.floor((ymax - ymin) / dy_eval)) + 1)
    x_domain = xmin + np.arange(Nx) * dx_eval
    y_domain = ymin + np.arange(Ny) * dy_eval

    # make sure the last point reaches the requested upper bound
    if x_domain[-1] < xmax:
        x_domain = np.append(x_domain, xmax)
    if y_domain[-1] < ymax:
        y_domain = np.append(y_domain, ymax)

    X_domain, Y_domain = np.meshgrid(x_domain, y_domain)
    return x_domain, y_domain, X_domain, Y_domain


# -------------------------------------------------------------------------
# Automatic seed box for the shrinklines
# -------------------------------------------------------------------------
def derive_saddle_band_from_near_axis_ftle(FTLE, x_domain, y_domain, band_half_height,
                                           profile_column_idx=1, drop_tolerance_fraction=1e-3):
    # Walk up the FTLE profile just off the axis and stop at the first clear peak.
    # That y-location sits near the rear saddle; we return a band around it.
    ftle_field = np.asarray(FTLE, dtype=float)
    x_coords = np.asarray(x_domain, dtype=float)
    y_coords = np.asarray(y_domain, dtype=float)
    if band_half_height <= 0:
        raise ValueError("band_half_height must be positive")

    order = np.argsort(y_coords)
    ys = y_coords[order]
    prof = ftle_field[order, profile_column_idx]
    good = np.isfinite(prof)
    ys = ys[good]
    prof = prof[good]
    if prof.size < 3 or np.allclose(prof, prof[0]):
        raise ValueError("Cannot find a peak in the near-axis FTLE profile")

    tol = max((prof.max() - prof.min()) * drop_tolerance_fraction, 1e-12)

    peak_idx = None
    min_seen = float(prof[0])
    for k in range(1, prof.size - 1):
        v = float(prof[k])
        has_risen = v > min_seen + tol
        has_turned = v >= prof[k - 1] - tol and v > prof[k + 1] + tol
        if has_risen and has_turned:
            peak_idx = k
            break
        min_seen = min(min_seen, v)

    mode = "first_local_max"
    if peak_idx is None:
        # no clean peak: fall back to the overall maximum
        peak_idx = int(np.argmax(prof))
        mode = "fallback_global_max"

    y_saddle = float(ys[peak_idx])
    y_band_min = float(max(y_coords.min(), y_saddle - band_half_height))
    y_band_max = float(min(y_coords.max(), y_saddle + band_half_height))
    return {
        "profile_x": float(x_coords[profile_column_idx]),
        "peak_detection_mode": mode,
        "peak_value": float(prof[peak_idx]),
        "y_saddle": y_saddle,
        "y_band_min": y_band_min,
        "y_band_max": y_band_max,
    }


def build_auto_seed_region(FTLE, x_domain, y_domain, band_half_height, seed_region_xbounds,
                           profile_column_idx=1):
    # Turn the FTLE peak band into a rectangular seed box [xmin, xmax, ymin, ymax].
    band = derive_saddle_band_from_near_axis_ftle(
        FTLE, x_domain, y_domain, band_half_height, profile_column_idx=profile_column_idx)
    seed_region = [seed_region_xbounds[0], seed_region_xbounds[1],
                   band["y_band_min"], band["y_band_max"]]
    return band, seed_region


def mask_eig_to_region(eig, X_domain, Y_domain, region):
    # Keep the eigenvalue field only inside the seed box (zero elsewhere) so the
    # seed search is limited to that box. region=None keeps the whole field.
    field = np.nan_to_num(np.asarray(eig, dtype=float), nan=0.0)
    if region is None:
        return field

    xmin, xmax, ymin, ymax = region
    field = field.copy()
    in_x = (X_domain[0, :] >= xmin) & (X_domain[0, :] <= xmax)
    in_y = (Y_domain[:, 0] >= ymin) & (Y_domain[:, 0] <= ymax)
    inside = np.outer(in_y, in_x)
    field[~inside] = 0.0
    print(f"  seed box [{xmin}, {xmax}] x [{ymin:.4f}, {ymax:.4f}]: "
          f"{np.count_nonzero(field)} grid points kept for seeding")
    return field


# -------------------------------------------------------------------------
# Tensorline integration from a given set of seeds
# -------------------------------------------------------------------------
def integrate_tensorlines_from_seeds(seed_x, seed_y, seed_val, X, Y, eig, vector_field,
                                     max_length, step_size, hyperbolicity=0, n_iterations=10**4):
    # Same tensorline loop as TBarrier, but the seeds are passed in instead of
    # being searched over the whole domain. The seeds come from the masked
    # (boxed) field, while the integration here uses the full eigenvalue field so
    # the lines are free to grow past the box.
    from math import sqrt
    from scipy.interpolate import RectBivariateSpline as RBS

    defined_domain = np.isfinite(eig).astype(int)
    eig_full = np.nan_to_num(np.asarray(eig, dtype=float), nan=0.0)
    interp_eig = RBS(Y[:, 0], X[0, :], eig_full, kx=1, ky=1)

    lines = [[], []]
    kept_x, kept_y, kept_val = [], [], []

    for i in range(len(seed_x)):
        if seed_val[i] <= hyperbolicity:
            continue

        forw = [[], []]
        back = [[], []]
        x = np.array([seed_x[i], seed_y[i]])
        forw[0].append(x[0])
        forw[1].append(x[1])

        vx, vy = _orient_vectorfield(X, Y, x, vector_field)
        xp_forw = np.array([vx, vy])
        xp_back = -np.array([vx, vy])
        x_forw = x.copy()
        x_back = x.copy()

        dist = 0
        counter = 0
        go_forw = True
        go_back = True
        while (go_forw or go_back) and counter < n_iterations:
            if go_forw and xp_forw is not None:
                x_forw, xp_forw = _RK4_tensorlines_incompressible(
                    X, Y, defined_domain, x_forw, xp_forw, step_size, vector_field, interp_eig)
                if x_forw is not None:
                    dist += sqrt(xp_forw[0] ** 2 + xp_forw[1] ** 2) * step_size
                    if dist < max_length:
                        forw[0].append(x_forw[0])
                        forw[1].append(x_forw[1])
                    else:
                        go_forw = False
                else:
                    go_forw = False

            if go_back and xp_back is not None:
                x_back, xp_back = _RK4_tensorlines_incompressible(
                    X, Y, defined_domain, x_back, xp_back, step_size, vector_field, interp_eig)
                if x_back is not None:
                    dist += sqrt(xp_back[0] ** 2 + xp_back[1] ** 2) * step_size
                    if dist < max_length:
                        back[0].append(x_back[0])
                        back[1].append(x_back[1])
                    else:
                        go_back = False
                else:
                    go_back = False

            counter += 1

        # stitch the backward half (reversed) onto the forward half
        line_x = np.append(np.flip(back[0]), forw[0])
        line_y = np.append(np.flip(back[1]), forw[1])
        lines[0].append(line_x)
        lines[1].append(line_y)
        kept_x.append(float(seed_x[i]))
        kept_y.append(float(seed_y[i]))
        kept_val.append(float(seed_val[i]))

    print(f"  integrated {len(kept_x)} line(s) from {len(seed_x)} seed(s)")
    return lines, np.asarray(kept_x), np.asarray(kept_y), np.asarray(kept_val)


# -------------------------------------------------------------------------
# FTLE field and Cauchy-Green eigen-data (run in parallel over the cores)
# -------------------------------------------------------------------------
def compute_ftle_eigen_parallel(X, Y, U, V, X_domain, Y_domain, time, time_data,
                                periodic, aux_grid_ratio=0.1, Ncores=4):
    # Returns FTLE, the two Cauchy-Green eigenvalues (eig1 = smaller, eig2 =
    # larger), their eigenvectors e1, e2, and R1, the final radial position of
    # each tracer (needed for the azimuthal check).
    lenT = abs(time[-1] - time[0])
    defined_domain = np.asarray(np.isfinite(U[:, :, 0]), dtype=int)
    dx = X_domain[0, 1] - X_domain[0, 0]
    dy = Y_domain[1, 0] - Y_domain[0, 0]
    aux_grid = [aux_grid_ratio * dx, aux_grid_ratio * dy]

    interp_u, interp_v = interpolant_unsteady(X, Y, U, V, method="cubic")

    def work(x0_batch, y0_batch):
        X0 = np.array([x0_batch, y0_batch])
        gradF = gradient_flowmap(time, X0, X, Y, interp_u, interp_v,
                                 periodic, defined_domain, True, time_data, aux_grid)
        gradF_final = gradF[-1, :, :, :]

        # one extra integration of the centre points to get the final radial position
        fmap = integration_dFdt(time, X0, X, Y, interp_u, interp_v, periodic, True, time_data)[0]
        r1 = fmap[-1, 0, :].astype(np.float64)

        npts = gradF_final.shape[2]
        ftle = np.full(npts, np.nan)
        l1 = np.full(npts, np.nan)
        l2 = np.full(npts, np.nan)
        v1 = np.full((npts, 2), np.nan)
        v2 = np.full((npts, 2), np.nan)
        for i in range(npts):
            dF = gradF_final[:, :, i]
            if not np.all(np.isfinite(dF)):
                continue
            C = CauchyGreen(dF)
            l1[i], l2[i], v1[i, :], v2[i, :] = eigen(C)
            ftle[i] = _FTLE_C(C, lenT)
        return ftle, l1, l2, v1, v2, r1

    x0 = X_domain.ravel()
    y0 = Y_domain.ravel()
    xb = split_evenly(x0, Ncores)
    yb = split_evenly(y0, Ncores)
    out = Parallel(n_jobs=Ncores, verbose=50)(
        delayed(work)(xb[i], yb[i]) for i in range(Ncores))

    ny, nx = X_domain.shape
    FTLE = np.concatenate([o[0] for o in out]).reshape(ny, nx)
    eig1 = np.concatenate([o[1] for o in out]).reshape(ny, nx)
    eig2 = np.concatenate([o[2] for o in out]).reshape(ny, nx)
    e1 = np.concatenate([o[3] for o in out], axis=0).reshape(ny, nx, 2)
    e2 = np.concatenate([o[4] for o in out], axis=0).reshape(ny, nx, 2)
    R1 = np.concatenate([o[5] for o in out]).reshape(ny, nx)
    return FTLE, eig1, eig2, e1, e2, R1


# -------------------------------------------------------------------------
# LAVD field (run in parallel over the cores)
# -------------------------------------------------------------------------
def compute_lavd_parallel(X, Y, U, V, X_domain, Y_domain, time, time_data,
                          periodic, aux_grid_ratio=0.1, Ncores=4):
    from ipynb.fs.defs.check_location import check_location

    if len(time) < 2:
        raise ValueError("LAVD needs at least two time levels")

    defined_domain = np.asarray(np.isfinite(U[:, :, 0]), dtype=int)
    dx_eval = float(X_domain[0, 1] - X_domain[0, 0])
    dy_eval = float(Y_domain[1, 0] - Y_domain[0, 0])
    aux_grid = [aux_grid_ratio * abs(dx_eval), aux_grid_ratio * abs(dy_eval)]
    dx_data = float(np.mean(np.diff(X[0, :])))
    dy_data = float(np.mean(np.diff(Y[:, 0])))
    delta_data = [abs(dx_data), abs(dy_data)]

    U_interp = np.nan_to_num(np.array(U, copy=True), nan=0.0)
    V_interp = np.nan_to_num(np.array(V, copy=True), nan=0.0)
    interp_u, interp_v = interpolant_unsteady(X, Y, U_interp, V_interp, method="cubic")

    def vorticity_batch(x0_batch, y0_batch):
        x0 = np.array([x0_batch, y0_batch])
        fmap = integration_dFdt(time, x0, X, Y, interp_u, interp_v, periodic, True, time_data)[0]

        # drop tracers that start outside the data
        for i in range(fmap.shape[2]):
            if check_location(X, Y, defined_domain, np.array([fmap[0, 0, i], fmap[0, 1, i]]))[0] != "IN":
                fmap[:, :, i] = np.nan

        omega = np.zeros((fmap.shape[0], fmap.shape[2]))
        for t in range(fmap.shape[0]):
            grad_vel = gradient_velocity(time[t], fmap[t, :, :], X, Y, interp_u, interp_v,
                                         periodic, defined_domain, True, time_data, delta_data, aux_grid)
            W = 0.5 * (grad_vel - grad_vel.transpose(1, 0, 2))
            omega[t, :] = 2.0 * W[0, 1, :]
        return omega

    x0 = X_domain.ravel()
    y0 = Y_domain.ravel()
    n_jobs = max(1, min(int(Ncores), len(x0)))
    xb = split_evenly(x0, n_jobs)
    yb = split_evenly(y0, n_jobs)
    out = Parallel(n_jobs=n_jobs, verbose=50)(
        delayed(vorticity_batch)(xb[i], yb[i]) for i in range(n_jobs))

    Omega = out[0]
    for o in out[1:]:
        Omega = np.append(Omega, o, axis=1)
    Omega = Omega.reshape((Omega.shape[0], X_domain.shape[0], X_domain.shape[1]))
    return _LAVD(Omega, np.asarray(time, dtype=float))


# -------------------------------------------------------------------------
# Axisymmetric check: in-plane stretch lambda1 versus azimuthal (R1/R0)^2
# -------------------------------------------------------------------------
def compute_lambda_theta(R1, X_domain, axis_tol=1e-9):
    R0 = X_domain
    with np.errstate(divide="ignore", invalid="ignore"):
        lambda_theta = (R1 / R0) ** 2
    lambda_theta = np.asarray(lambda_theta, dtype=float)
    lambda_theta[R0 <= axis_tol] = np.nan   # undefined on the axis
    return lambda_theta


def sample_field_along_lines(field, x_domain, y_domain, lines_x, lines_y):
    # Interpolate a grid field onto each polyline.
    interp = RegularGridInterpolator((y_domain, x_domain), field,
                                     bounds_error=False, fill_value=np.nan)
    sampled = []
    for lx, ly in zip(lines_x, lines_y):
        pts = np.column_stack([np.asarray(ly, float).ravel(), np.asarray(lx, float).ravel()])
        sampled.append(interp(pts))
    return sampled


def azimuthal_fraction_along_lines(eig2, lambda_theta, x_domain, y_domain, lines_x, lines_y):
    # Fraction of line points where the in-plane stretch beats the azimuthal one.
    l1 = sample_field_along_lines(eig2, x_domain, y_domain, lines_x, lines_y)
    lt = sample_field_along_lines(lambda_theta, x_domain, y_domain, lines_x, lines_y)
    cat_l1 = np.concatenate([np.ravel(a) for a in l1]) if l1 else np.array([])
    cat_lt = np.concatenate([np.ravel(a) for a in lt]) if lt else np.array([])
    ok = np.isfinite(cat_l1) & np.isfinite(cat_lt)
    frac = float(np.mean(cat_l1[ok] > cat_lt[ok])) if ok.any() else np.nan
    return frac, l1, lt


# -------------------------------------------------------------------------
# Building the list of runs and naming the output files
# -------------------------------------------------------------------------
def as_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return [value]


def build_run_specs(start_snapshot_idx, time_interval_snapshots, dx_eval, dy_eval):
    # Each of the inputs may be a single value or a list. Pair them up and return
    # one dict per (time window) x (grid spacing) combination.
    starts = [int(v) for v in as_list(start_snapshot_idx)]
    intervals = [int(v) for v in as_list(time_interval_snapshots)]
    dxs = [float(v) for v in as_list(dx_eval)]
    dys = [float(v) for v in as_list(dy_eval)]

    if len(starts) == 1 and len(intervals) > 1:
        starts = starts * len(intervals)
    elif len(intervals) == 1 and len(starts) > 1:
        intervals = intervals * len(starts)
    elif len(starts) != len(intervals):
        raise ValueError("start and interval lists must match in length (or one be a single value)")

    if len(dxs) == 1 and len(dys) > 1:
        dxs = dxs * len(dys)
    elif len(dys) == 1 and len(dxs) > 1:
        dys = dys * len(dxs)
    elif len(dxs) != len(dys):
        raise ValueError("dx and dy lists must match in length (or one be a single value)")

    runs = []
    for s, it in zip(starts, intervals):
        if it == 0:
            raise ValueError("time interval cannot be zero")
        for dx, dy in zip(dxs, dys):
            if dx <= 0 or dy <= 0:
                raise ValueError("grid spacing must be positive")
            runs.append({
                "start_snapshot_idx": s,
                "end_snapshot_idx": s + it,
                "time_interval_snapshots": it,
                "dx_eval": dx,
                "dy_eval": dy,
            })
    return runs


def grid_run_tag(folder_name, integration_dt, dx_eval, dy_eval):
    return f"{folder_name}_idt{integration_dt:g}_dxy{dx_eval:g}x{dy_eval:g}"


def min_distance_tag(min_distance):
    return f"md{min_distance:g}".replace("-", "m").replace(".", "p")


def build_output_path(desired_folder, run_folder_name, pipeline):
    out = os.path.join(lcs_rising_bubble_root(), 'Run', pipeline, run_folder_name)
    if desired_folder:
        out = os.path.join(out, str(desired_folder))
    os.makedirs(out, exist_ok=True)
    return out


# -------------------------------------------------------------------------
# Saving .mat and .png
# -------------------------------------------------------------------------
def lines_to_object_arrays(lines_x, lines_y):
    # Lines have different lengths, so store them as a MATLAB cell array.
    n = len(lines_x)
    ox = np.empty(n, dtype=object)
    oy = np.empty(n, dtype=object)
    for i in range(n):
        ox[i] = np.asarray(lines_x[i], dtype=float)
        oy[i] = np.asarray(lines_y[i], dtype=float)
    return ox, oy


def save_ftle_outputs(output_path, run_tag, field_prefix, start_idx, end_idx, md,
                      payload, X_domain, Y_domain, FTLE, shrinklines, stretchlines,
                      X, Y, PHI_snap, xmin, xmax, ymin, ymax, title_start, title_end):
    import scipy.io
    import matplotlib.pyplot as plt

    name = f"{run_tag}_{min_distance_tag(md)}_{field_prefix}_{start_idx}_{end_idx}"
    scipy.io.savemat(os.path.join(output_path, name + ".mat"), payload)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=600)
    ax.contourf(X_domain, Y_domain, np.ma.masked_invalid(FTLE), cmap="turbo", levels=600)
    if shrinklines is not None:
        for i in range(len(shrinklines[0])):
            ax.plot(shrinklines[0][i], shrinklines[1][i], c='r', lw=0.8, ls='dashed')
    if stretchlines is not None:
        for i in range(len(stretchlines[0])):
            ax.plot(stretchlines[0][i], stretchlines[1][i], c='b', lw=0.8, ls='dashed')
    if PHI_snap is not None:
        ax.contour(X, Y, PHI_snap, levels=[0], colors='b', linewidths=1.2)
    ax.set_xlabel("r", fontsize=8)
    ax.set_ylabel("z", fontsize=8)
    ax.set_title(rf'$\mathrm{{FTLE}}_{{{title_start}}}^{{{title_end}}}$', fontsize=8)
    ax.set_aspect('equal')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.savefig(os.path.join(output_path, name + ".png"), dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {name}")


def save_lavd_outputs(output_path, run_tag, start_idx, end_idx, payload,
                      X_domain, Y_domain, LAVD, vortices, center_x, center_y,
                      xmin, xmax, ymin, ymax):
    import scipy.io
    import matplotlib.pyplot as plt

    name = f"{run_tag}_LAVD_{start_idx}_{end_idx}"
    scipy.io.savemat(os.path.join(output_path, name + ".mat"), payload)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=600)
    ax.contourf(X_domain, Y_domain, np.ma.masked_invalid(LAVD), cmap="turbo", levels=600)
    if vortices is not None:
        for v in vortices:
            if v is not None and len(v) >= 2:
                ax.plot(v[0], v[1], c='b', lw=1.0)
    if len(center_x):
        ax.scatter(center_x, center_y, s=8, c='k')
    ax.set_xlabel("r", fontsize=8)
    ax.set_ylabel("z", fontsize=8)
    ax.set_title("LAVD", fontsize=8)
    ax.set_aspect('equal')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.savefig(os.path.join(output_path, name + ".png"), dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {name}")


def vortices_to_object_arrays(vortices, center_x, center_y):
    # Pack vortex boundaries and centres as cell arrays for MATLAB.
    n = max(len(vortices), len(center_x))
    vx = np.empty(n, dtype=object)
    vy = np.empty(n, dtype=object)
    cx = np.empty(n, dtype=object)
    cy = np.empty(n, dtype=object)
    for i in range(n):
        if i < len(vortices) and vortices[i] is not None and len(vortices[i]) >= 2:
            vx[i] = np.asarray(vortices[i][0])
            vy[i] = np.asarray(vortices[i][1])
        else:
            vx[i] = np.array([np.nan])
            vy[i] = np.array([np.nan])
        if i < len(center_x):
            cx[i] = np.asarray([center_x[i]], dtype=np.float64)
            cy[i] = np.asarray([center_y[i]], dtype=np.float64)
        else:
            cx[i] = np.array([np.nan])
            cy[i] = np.array([np.nan])
    return vx, vy, cx, cy
