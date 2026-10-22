import numpy as np

# =============================== inputs ===============================
folder_name = "3_2D_Ga60_Bo65_SD0_0_DR100_VR100"   # case folder inside database/
Flip_Snapshot = True                                # flip [nx, ny] -> [ny, nx] when reading
snapshot_file_offset = 0                             # offset added to the snapshot file numbers
snapshot_dt = 0.05                                  # time between two stored snapshots
start_snapshot_idx = 178                            # first snapshot (single value or a list)
time_interval_snapshots = 400                       # length of the window (negative = backward FTLE)
dx_eval = 0.00625                                   # tracer spacing in x (single value or a list)
dy_eval = 0.00625                                   # tracer spacing in y
Ncores = 26                                         # number of cores
periodic = [False, False, False]                    # periodic in [x, y, time]
desired_folder = "TBarrier_modified_draft"          # output subfolder under Run/FTLE2D/<case>/ (None = none)

# shrinkline / stretchline settings
LCS_type = [1, 0]                                   # compute [shrinklines, stretchlines]
min_distance_values = [0.1]                         # min distance between seeds (single value or a list)
max_length = 2.4                                    # longest a line is allowed to grow
n_tensorlines = -1                                  # -1 = use every retained seed
hyperbolicity = 0.0                                 # minimum eigenvalue needed to keep a seed
arc_length_budget = 2.4                             # sets the number of integration steps
step_size_ratio = 0.05                              # step_size = step_size_ratio * dx_eval
aux_grid_ratio = 0.1                                # auxiliary grid spacing for the flow-map gradient

# FTLE box: the top follows the bubble, the height is fixed
auto_domain_from_bubble_top = True
xmin = 0.0
xmax = 1.0
L_y = 3.0                                           # box height; ymin = bubble_top + padding - L_y
auto_domain_top_padding = 0.2                       # ymax = bubble_top + this
manual_ymin = None                                  # used only if auto_domain_from_bubble_top = False
manual_ymax = None

# seed box: y-band from the first FTLE peak off the axis, x-width set here
auto_saddle_band_half_height = 0.05
seed_region_xbounds = [0.0, 0.2]

# ======================================================================
import math
import rising_bubble_extensions as ext
from ipynb.fs.defs.loc_max import _loc_max   # TBarrier seed finder (applies min_distance)


runs = ext.build_run_specs(start_snapshot_idx, time_interval_snapshots, dx_eval, dy_eval)

if isinstance(min_distance_values, (list, tuple, np.ndarray)):
    md_values = [float(v) for v in min_distance_values]
else:
    md_values = [float(min_distance_values)]

for run in runs:
    run_start = int(run["start_snapshot_idx"])
    run_end = int(run["end_snapshot_idx"])
    run_dx = float(run["dx_eval"])
    run_dy = float(run["dy_eval"])

    read_folder = ext.resolve_database_folder(folder_name)
    output_path = ext.build_output_path(desired_folder, folder_name, "FTLE2D")

    # load grid and fields
    XX, YY = ext.load_grid(read_folder)
    nx, ny = len(XX), len(YY)
    X, Y = np.meshgrid(XX, YY)
    U, V = ext.load_velocity_snapshots(read_folder, snapshot_file_offset, run_start, run_end, nx, ny, Flip_Snapshot)
    PHI = ext.load_phi_snapshots(read_folder, snapshot_file_offset, run_start, run_end, nx, ny, Flip_Snapshot)

    # build the time axis; the sign of the step sets forward vs backward FTLE
    direction = 1 if run_end >= run_start else -1
    integration_dt = direction * snapshot_dt
    lower = min(run_start, run_end)
    upper = max(run_start, run_end)
    nt = upper - lower + 1
    t0 = run_start * snapshot_dt
    tN = run_end * snapshot_dt
    time_data = np.reshape(np.linspace(lower * snapshot_dt, upper * snapshot_dt, nt), (1, -1))
    time = np.arange(t0, tN + integration_dt, integration_dt)
    field_prefix = "bFTLE" if integration_dt < 0 else "fFTLE"
    phi_eval_idx = 0 if direction == 1 else -1
    PHI_snap = PHI[:, :, phi_eval_idx]

    # evaluation box
    if auto_domain_from_bubble_top:
        phi_start_idx = run_start - lower
        bubble_top_y, ymin, ymax = ext.derive_ftle_y_bounds_from_bubble(
            PHI[:, :, phi_start_idx], Y, auto_domain_top_padding, L_y)
        print(f"box: bubble_top={bubble_top_y:.4f}, ymin={ymin:.4f}, ymax={ymax:.4f}")
    else:
        if manual_ymin is None or manual_ymax is None:
            raise ValueError("set manual_ymin/manual_ymax when auto_domain_from_bubble_top = False")
        ymin, ymax = float(manual_ymin), float(manual_ymax)

    x_domain, y_domain, X_domain, Y_domain = ext.build_ftle_domain(
        xmin, xmax, ymin, ymax, run_dx, run_dy, X, Y, t0, tN, time_data)

    # FTLE field + eigen-data + final radial position
    FTLE, eig1, eig2, e1, e2, R1 = ext.compute_ftle_eigen_parallel(
        X, Y, U, V, X_domain, Y_domain, time, time_data, periodic,
        aux_grid_ratio=aux_grid_ratio, Ncores=Ncores)

    # axisymmetric check: in-plane stretch (eig2) vs azimuthal (R1/R0)^2
    lambda_theta = ext.compute_lambda_theta(R1, X_domain)
    valid = np.isfinite(eig2) & np.isfinite(lambda_theta)
    lambda1_gt_theta = np.full(eig2.shape, np.nan)
    lambda1_gt_theta[valid] = (eig2[valid] > lambda_theta[valid]).astype(float)

    # seed box from the near-axis FTLE peak
    band, seed_region = ext.build_auto_seed_region(
        FTLE, x_domain, y_domain, auto_saddle_band_half_height, seed_region_xbounds)
    print(f"seed band: y_saddle={band['y_saddle']:.4f}, FTLE_peak={band['peak_value']:.4f}, mode={band['peak_detection_mode']}")

    step_size = step_size_ratio * run_dx
    n_iterations = math.ceil(arc_length_budget / step_size)
    run_tag = ext.grid_run_tag(folder_name, integration_dt, run_dx, run_dy)

    for md in md_values:
        print(f"case {folder_name}  {run_start}->{run_end}  grid {run_dx:g}x{run_dy:g}  "
              f"min_distance={md:g}  step={step_size:g}  iters={n_iterations}")

        shrinklines = stretchlines = None
        seed_x = seed_y = seed_val = np.array([])
        sl_lambda1 = sl_lambda_theta = None
        shrink_frac = np.nan

        # seeds only inside the box; min_distance keeps the strongest one or two
        eig2_masked = ext.mask_eig_to_region(eig2, X_domain, Y_domain, seed_region)

        if LCS_type[0]:
            _, _, seed_x, seed_y, seed_val = _loc_max(md, X_domain, Y_domain, eig2_masked)
            shrinklines, seed_x, seed_y, seed_val = ext.integrate_tensorlines_from_seeds(
                seed_x, seed_y, seed_val, X_domain, Y_domain, eig2, e1,
                max_length, step_size, hyperbolicity, n_iterations)
            shrink_frac, sl_lambda1, sl_lambda_theta = ext.azimuthal_fraction_along_lines(
                eig2, lambda_theta, x_domain, y_domain, shrinklines[0], shrinklines[1])
            print(f"azimuthal check on shrinklines: in-plane wins at {100 * shrink_frac:.1f}% of points")

        if LCS_type[1]:
            _, _, st_x, st_y, st_val = _loc_max(md, X_domain, Y_domain, eig2_masked)
            stretchlines, _, _, _ = ext.integrate_tensorlines_from_seeds(
                st_x, st_y, st_val, X_domain, Y_domain, eig2, e2,
                max_length, step_size, hyperbolicity, n_iterations)

        # collect everything for the .mat file
        payload = {
            "FTLE": FTLE, "x": x_domain, "y": y_domain,
            "eig1": eig1, "eig2": eig2, "R1_final": R1,
            "lambda_theta": lambda_theta, "lambda1_gt_lambda_theta": lambda1_gt_theta,
            "min_distance": np.array([[md]], dtype=np.float64),
            "dx_eval": np.array([[run_dx]], dtype=np.float64),
            "dy_eval": np.array([[run_dy]], dtype=np.float64),
            "start_snapshot_idx": np.array([[run_start]], dtype=np.int64),
            "end_snapshot_idx": np.array([[run_end]], dtype=np.int64),
            "time_interval_snapshots": np.array([[run["time_interval_snapshots"]]], dtype=np.int64),
            "seed_region": np.array(seed_region, dtype=np.float64),
            "seed_x": np.asarray(seed_x, dtype=float),
            "seed_y": np.asarray(seed_y, dtype=float),
            "seed_val": np.asarray(seed_val, dtype=float),
        }
        if shrinklines is not None:
            sl_x, sl_y = ext.lines_to_object_arrays(shrinklines[0], shrinklines[1])
            payload["shrinklines_x"] = sl_x
            payload["shrinklines_y"] = sl_y
            payload["shrinklines_check_frac"] = np.array([[shrink_frac]], dtype=np.float64)
            if sl_lambda1 is not None:
                l1_obj = np.empty(len(sl_lambda1), dtype=object)
                lt_obj = np.empty(len(sl_lambda_theta), dtype=object)
                for i in range(len(sl_lambda1)):
                    l1_obj[i] = np.asarray(sl_lambda1[i])
                    lt_obj[i] = np.asarray(sl_lambda_theta[i])
                payload["shrinklines_lambda1"] = l1_obj
                payload["shrinklines_lambda_theta"] = lt_obj
        if stretchlines is not None:
            st_x_obj, st_y_obj = ext.lines_to_object_arrays(stretchlines[0], stretchlines[1])
            payload["stretchlines_x"] = st_x_obj
            payload["stretchlines_y"] = st_y_obj

        ext.save_ftle_outputs(
            output_path, run_tag, field_prefix, run_start, run_end, md, payload,
            X_domain, Y_domain, FTLE, shrinklines, stretchlines,
            X, Y, PHI_snap, xmin, xmax, ymin, ymax, run_start, run_end)
