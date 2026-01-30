import numpy as np

from geometry.pose import matrix_to_quat, flat_to_matrix
from geometry.pose import euler_to_quat, quat_to_matrix

from scripts.object_cloud_point import ObjectPointCloud


########## Sinusoidal functions ##########
def get_sin_velocity_profile_peak(dist, duration):
    """Get the peak speed and acceleration for a sinusoidal velocity profile"""
    # v_max = 2 * dist / duration
    # a_max = v_max * np.pi / duration
    peak_speed = 2 * dist / duration
    peak_acc = peak_speed * np.pi / duration
    return peak_speed, peak_acc


def sin_velocity_profile(t, dist, duration):
    """Get the velocity at time t for a given push distance d and duration."""
    v_max, acc_max = get_sin_velocity_profile_peak(dist, duration)  # (N, 1)
    # Compute the velocity at time t for each sample given a sin velocity
    # v = (v_max / 2) * (np.sin(2 * np.pi * t / duration - np.pi / 2) + 1)
    # a = acc_max * np.sin(2 * np.pi * t / duration)
    scale = -v_max * duration / 4 / np.pi  # (N,)
    # get the position d at given time t (N, T)
    d = scale * np.sin(2 * np.pi * t / duration) + (v_max * t / 2)
    return d


########## Geometry functions ##########
def euler_to_matrix(seq: str, euler: np.ndarray) -> np.ndarray:
    """Convert Euler angles to rotation matrix"""
    quat = euler_to_quat(euler, seq)
    matrix = quat_to_matrix(quat)
    return matrix


def get_local_curvature(points, window_size):
    num_points = len(points)
    curvatures = np.zeros(num_points)
    points_xy = points[:, :2]
    half_win = window_size // 2

    for i in range(num_points):
        indices = np.arange(i - half_win, i + half_win + 1) % num_points
        neighborhood = points_xy[indices]

        x = neighborhood[:, 0]
        y = neighborhood[:, 1]

        A = np.column_stack([2*x, 2*y, np.ones(len(x))])
        B = x**2 + y**2

        try:
            res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            a, b, C = res
            
            # 4. Calculate Radius and Curvature
            R_squared = C + a**2 + b**2
            if R_squared <= 0:
                curvatures[i] = 0 # Numerical error or straight line
            else:
                R = np.sqrt(R_squared)
                # Curvature is 1/R. We cap it to avoid infinity on flat surfaces
                curvatures[i] = 1.0 / R if R > 1e-6 else 0
        except np.linalg.LinAlgError:
            curvatures[i] = 0

    return curvatures


def get_local_PCA(points, window_size=5):
    num_points = len(points)
    curvatures = np.zeros(num_points)
    points_xy = points[:, :2]
    half_win = window_size // 2

    for i in range(num_points):
        indices = np.arange(i - half_win, i + half_win + 1) % num_points
        neighborhood = points_xy[indices]
        
        # Center the neighborhood
        centered = neighborhood - np.mean(neighborhood, axis=0)
        
        # Covariance matrix
        cov = np.dot(centered.T, centered) / len(neighborhood)
        
        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov)
        
        # Surface variation: small_eigenvalue / sum_of_eigenvalues
        # For a line, eigenvalues are [0, big_val] -> curvature 0
        # For a circle, eigenvalues are [val, val] -> curvature 0.5
        sum_eigen = np.sum(eigenvalues)
        if sum_eigen > 1e-9:
            curvatures[i] = eigenvalues[0] / sum_eigen
        else:
            curvatures[i] = 0
            
    return curvatures


def get_cloud_point(obj_name, n_params):
    mesh_path = f"assets/{obj_name}/textured.obj"
    pcd = ObjectPointCloud(
        mesh_path,
        z_height=0.0,
        num_points=n_params,
        thickness=0.001,
        sort_by_angle=True
    )
    return pcd.points, pcd.normals, pcd.angles


def get_cloud_point_push(
    n_params: int,
    obj_states: np.ndarray,
    obj_shape: tuple[float, float, float],
    obj_name: str,
    tool_offset: np.ndarray = np.array([0, 0, 0, 1, 0, 0, 0]),
    pre_push_offset: float = 0.02,
    duration: float = 2,
    dt: float = 0.1,
    max_speed: float = 0.5,
    max_acc: float = 1,
):
    # Get cloud points [N_points, 3], normals [N_points, 3], angles [N_points,]
    points, normals, angles = get_cloud_point(obj_name, n_params)

    # Calculate Curvature [N_points,]
    curvatures = get_local_curvature(points, window_size=10)
    # curvatures = get_local_PCA(points, window_size=50)

    # Reshape angles to [N_points, 1] for concatenation
    angles_reshaped = angles.reshape(-1, 1)
    curvatures_reshaped = curvatures.reshape(-1, 1)

    # Concatenate: points(3) + normals(3) + angles(1) + curvature(1) = 8
    push_params = np.concatenate([points, normals, angles_reshaped, curvatures_reshaped], axis=1)    # [N_points, 8]

    # Generate paths
    # times, ws_paths = generate_center_push_path(
    times, ws_paths = generate_normal_push_path(
        obj_states,
        obj_shape,
        push_params,
        tool_offset,
        pre_push_offset,
        duration,
        dt,
        max_speed,
        max_acc,
    )
    return push_params, times, ws_paths


def generate_center_push_path(
    obj_states: np.ndarray,
    obj_shape: tuple[float, float, float],
    push_params: np.ndarray,
    tool_offset: np.ndarray = np.array([0, 0, 0, 1, 0, 0, 0]),
    pre_push_offset: float = 0.02,
    duration: float = 2,
    dt: float = 0.1,
    max_speed: float = 0.5,
    max_acc: float = 1,
):
    """Generate workspace paths for radial pushes toward the center"""
    # push_params [x, y, z, nx, ny, nz, angle, curvature]
    assert push_params.ndim == 2 and push_params.shape[1] == 8, f"Expected push_params shape (N, 8), got {push_params.shape}"
    
    n_data = push_params.shape[0]
    
    # Unpack data
    contact_points = push_params[:, :3]      # [N, 3] - (x, y, z)
    contact_normals = push_params[:, 3:6]    # [N, 3] - (nx, ny, nz)
    angles = push_params[:, 6]               # [N,] - angle in radians
    curvatures = push_params[:, 7]           # [N,] - curvatures
    
    # Calculate Object "Radius" (Max extent from center)
    length_x, length_y, _ = obj_shape
    obj_max_r = np.sqrt(length_x**2 + length_y**2) / 2.0

    # 1. Direction Vectors (towards center)
    push_dir_x = -np.cos(angles)
    push_dir_y = -np.sin(angles)
    dir_vecs = np.stack([push_dir_x, push_dir_y], axis=1) # (N, 2)

    # 2. Start Points
    start_dist = obj_max_r + pre_push_offset
    starts = np.stack([np.cos(angles), np.sin(angles)], axis=1) * start_dist

    # 3. Velocity Profile
    dist_scalar = start_dist * 2
    total_push_dists = np.full((n_data, 1), dist_scalar)

    peak_speed, peak_acc = get_sin_velocity_profile_peak(total_push_dists, duration)
    assert np.all(peak_speed <= max_speed) and np.all(peak_acc <= max_acc)
    
    n_steps = int(duration / dt)
    t_paths = np.tile(np.linspace(0, duration, n_steps), (n_data, 1))

    path_dists = sin_velocity_profile(t_paths, total_push_dists, duration)

    # 4. Local Path
    local_xy = starts[:, None, :] + path_dists[:, :, None] * dir_vecs[:, None, :]
    local_z = np.zeros((n_data, n_steps, 1))
    local_pos = np.concatenate([local_xy, local_z], axis=2)

    # 5. End Effector Orientation
    # Tool faces the center (angle + PI)
    t_rotate_z = np.tile(np.eye(4)[None, None, :, :], (n_data, 1, 1, 1))

    for i in range(n_data):
        t_rotate_z[i, 0, :3, :3] = euler_to_matrix("z", angles[i] + np.pi)

    t_reflect_z = np.eye(4)[None, None, :, :]
    t_reflect_z[0, 0, :3, :3] = euler_to_matrix("x", np.pi) # Pointing down
    t_delta = t_rotate_z @ t_reflect_z @ flat_to_matrix(tool_offset)[None, None, :, :]

    # 6. Global Transform
    t_local_pos = np.tile(np.eye(4)[None, None, :, :], (n_data, n_steps, 1, 1))
    t_local_pos[:, :, :3, 3] = local_pos
    t_global = flat_to_matrix(obj_states)[:, None, :, :] @ (t_local_pos @ t_delta)

    # 7. Flatten to 7D
    ws_pos = t_global[:, :, :3, 3]
    ws_quat = matrix_to_quat(t_global[:, :, :3, :3].reshape(-1, 3, 3))
    ws_quat = ws_quat.reshape(n_data, n_steps, 4)
    ws_paths = np.concatenate([ws_pos, ws_quat], axis=-1)

    return t_paths, ws_paths


def generate_normal_push_path(
    obj_states: np.ndarray,
    obj_shape: tuple[float, float, float],
    push_params: np.ndarray,
    tool_offset: np.ndarray = np.array([0, 0, 0, 1, 0, 0, 0]),
    pre_push_offset: float = 0.02,
    duration: float = 2,
    dt: float = 0.1,
    max_speed: float = 0.5,
    max_acc: float = 1,
):
    """Generate workspace paths for radial pushes toward the center"""
    # push_params [x, y, z, nx, ny, nz, angle, curvature]
    assert push_params.ndim == 2 and push_params.shape[1] == 8, f"Expected push_params shape (N, 8), got {push_params.shape}"
    
    n_data = push_params.shape[0]
    
    # Unpack data
    contact_points = push_params[:, :3]      # [N, 3] - (x, y, z)
    contact_normals = push_params[:, 3:6]    # [N, 3] - (nx, ny, nz)
    angles = push_params[:, 6]               # [N,] - angle in radians
    curvatures = push_params[:, 7]           # [N,] - curvatures
    
    # Calculate Object "Radius" (Max extent from center)
    length_x, length_y, _ = obj_shape
    obj_max_r = np.sqrt(length_x**2 + length_y**2) / 2.0
    
    # 1. Direction Vectors (opposite to x,y normals = inward)
    dir_vecs = -contact_normals[:, :2]

    # Normalize direction vectors
    dir_vecs = dir_vecs / (np.linalg.norm(dir_vecs, axis=1, keepdims=True) + 1e-8)

    # 2. Start Points (start at contact point, offset outward along the normal)
    start_dist = obj_max_r + pre_push_offset
    starts = contact_points[:, :2] + pre_push_offset * contact_normals[:, :2]

    # 3. Velocity Profile
    dist_scalar = start_dist * 2
    total_push_dists = np.full((n_data, 1), dist_scalar)

    peak_speed, peak_acc = get_sin_velocity_profile_peak(total_push_dists, duration)
    assert np.all(peak_speed <= max_speed) and np.all(peak_acc <= max_acc)

    n_steps = int(duration / dt)
    t_paths = np.tile(np.linspace(0, duration, n_steps), (n_data, 1))

    path_dists = sin_velocity_profile(t_paths, total_push_dists, duration)
    
    # 4. Local Path
    local_xy = starts[:, None, :] + path_dists[:, :, None] * dir_vecs[:, None, :]
    local_z = np.zeros((n_data, n_steps, 1))
    local_pos = np.concatenate([local_xy, local_z], axis=2)
    
    # 5. End Effector Orientation
    # The tool should be aligned with the push direction (normal)
    normal_angles = np.arctan2(contact_normals[:, 1], contact_normals[:, 0])

    t_rotate_z = np.tile(np.eye(4)[None, None, :, :], (n_data, 1, 1, 1))
    
    for i in range(n_data):
        t_rotate_z[i, 0, :3, :3] = euler_to_matrix("z", normal_angles[i] + np.pi)
    
    t_reflect_z = np.eye(4)[None, None, :, :]
    t_reflect_z[0, 0, :3, :3] = euler_to_matrix("x", np.pi) # Pointing down
    t_delta = t_rotate_z @ t_reflect_z @ flat_to_matrix(tool_offset)[None, None, :, :]

    # 6. Global Transform
    t_local_pos = np.tile(np.eye(4)[None, None, :, :], (n_data, n_steps, 1, 1))
    t_local_pos[:, :, :3, 3] = local_pos
    t_global = flat_to_matrix(obj_states)[:, None, :, :] @ (t_local_pos @ t_delta)

    # 7. Flatten to 7D
    ws_pos = t_global[:, :, :3, 3]
    ws_quat = matrix_to_quat(t_global[:, :, :3, :3].reshape(-1, 3, 3))
    ws_quat = ws_quat.reshape(n_data, n_steps, 4)
    ws_paths = np.concatenate([ws_pos, ws_quat], axis=-1)

    return t_paths, ws_paths
