import numpy as np

# =============================== inputs ===============================
folder_name = "3_2D_Ga60_Bo65_SD0_0_DR100_VR100"   # case folder inside database/
Flip_Snapshot = True                                # flip [nx, ny] -> [ny, nx] when reading
snapshot_file_offset = 0                             # offset added to the snapshot file numbers
snapshot_dt = 0.05                                  # time between two stored snapshots
start_snapshot_idx = 60                             # first snapshot
end_snapshot_idx = 330                              # last snapshot
dx_eval = 0.00625                                   # tracer spacing in x
dy_eval = 0.00625                                   # tracer spacing in y
Ncores = 28                                         # number of cores
periodic = [False, False, False]                    # periodic in [x, y, time]
desired_folder = "TBarrier_modified_draft"          # output subfolder under Run/LAVD2D/<case>/ (None = none)

# vortex detection
extract_vortices = True                             # find vortex centres and outer boundaries
vortex_center_min_distance = 0.1                    # min distance between LAVD maxima
contour_level_count = 600                           # how many level sets to test per maximum
convexity_deficiency = 1e-2                         # how far from convex a boundary may be
boundary_arc_length_min = 0.1                       # shortest boundary we accept
loc_threshold_fraction = 0.05                       # ignore maxima below this fraction of max(LAVD)
aux_grid_ratio = 0.1                                # auxiliary grid spacing for the velocity gradient

# LAVD box: the top follows the bubble, the height is fixed
auto_domain_from_bubble_top = True
xmin = 0.0
xmax = 2.0
L_y = 3.0
auto_domain_top_padding = 0.2
manual_ymin = None
manual_ymax = None

# ======================================================================
import rising_bubble_extensions as ext
from ipynb.fs.defs.loc_max import _loc_max
from ipynb.fs.defs.find_outermost_contour import find_outermost_contour


read_folder = ext.resolve_database_folder(folder_name)
output_path = ext.build_output_path(desired_folder, folder_name, "LAVD2D")

# load grid and fields
XX, YY = ext.load_grid(read_folder)
nx, ny = len(XX), len(YY)
X, Y = np.meshgrid(XX, YY)
U, V = ext.load_velocity_snapshots(read_folder, snapshot_file_offset, start_snapshot_idx, end_snapshot_idx, nx, ny, Flip_Snapshot)
PHI = ext.load_phi_snapshots(read_folder, snapshot_file_offset, start_snapshot_idx, end_snapshot_idx, nx, ny, Flip_Snapshot)

# time axis
direction = 1 if end_snapshot_idx >= start_snapshot_idx else -1
integration_dt = direction * snapshot_dt
lower = min(start_snapshot_idx, end_snapshot_idx)
upper = max(start_snapshot_idx, end_snapshot_idx)
nt = upper - lower + 1
t0 = start_snapshot_idx * snapshot_dt
tN = end_snapshot_idx * snapshot_dt
time_data = np.reshape(np.linspace(lower * snapshot_dt, upper * snapshot_dt, nt), (1, -1))
time = np.arange(t0, tN + integration_dt, integration_dt)

# evaluation box
if auto_domain_from_bubble_top:
    phi_start_idx = start_snapshot_idx - lower
    bubble_top_y, ymin, ymax = ext.derive_ftle_y_bounds_from_bubble(
        PHI[:, :, phi_start_idx], Y, auto_domain_top_padding, L_y)
    print(f"box: bubble_top={bubble_top_y:.4f}, ymin={ymin:.4f}, ymax={ymax:.4f}")
else:
    if manual_ymin is None or manual_ymax is None:
        raise ValueError("set manual_ymin/manual_ymax when auto_domain_from_bubble_top = False")
    ymin, ymax = float(manual_ymin), float(manual_ymax)

x_domain, y_domain, X_domain, Y_domain = ext.build_ftle_domain(
    xmin, xmax, ymin, ymax, dx_eval, dy_eval, X, Y, t0, tN, time_data)

# LAVD field
LAVD = ext.compute_lavd_parallel(
    X, Y, U, V, X_domain, Y_domain, time, time_data, periodic,
    aux_grid_ratio=aux_grid_ratio, Ncores=Ncores)

# vortex centres (LAVD maxima) and their outermost convex boundaries
center_x = center_y = center_val = np.array([])
vortices = []
if extract_vortices:
    loc_threshold = loc_threshold_fraction * float(np.nanmax(LAVD))
    _, _, center_x, center_y, center_val = _loc_max(
        vortex_center_min_distance, X_domain, Y_domain, LAVD, loc_threshold)
    vortices = find_outermost_contour(
        X_domain, Y_domain, LAVD, vortex_center_min_distance, contour_level_count,
        convexity_deficiency, boundary_arc_length_min, loc_threshold, Ncores)

# collect everything for the .mat file
vortex_x, vortex_y, vcx, vcy = ext.vortices_to_object_arrays(vortices, center_x, center_y)
payload = {
    "LAVD": LAVD, "x": x_domain, "y": y_domain,
    "dx_eval": np.array([[dx_eval]], dtype=np.float64),
    "dy_eval": np.array([[dy_eval]], dtype=np.float64),
    "start_snapshot_idx": np.array([[start_snapshot_idx]], dtype=np.int64),
    "end_snapshot_idx": np.array([[end_snapshot_idx]], dtype=np.int64),
    "vortex_x": vortex_x, "vortex_y": vortex_y,
    "vortex_center_x": vcx, "vortex_center_y": vcy,
    "center_val": np.asarray(center_val, dtype=float),
}

run_tag = ext.grid_run_tag(folder_name, integration_dt, dx_eval, dy_eval)
ext.save_lavd_outputs(
    output_path, run_tag, start_snapshot_idx, end_snapshot_idx, payload,
    X_domain, Y_domain, LAVD, vortices, center_x, center_y, xmin, xmax, ymin, ymax)
