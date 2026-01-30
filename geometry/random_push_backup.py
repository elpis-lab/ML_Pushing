import numpy as np

from geometry.pose import matrix_to_quat, flat_to_matrix
from geometry.pose import euler_to_quat, quat_to_matrix

from scripts.object_cloud_point import ObjectPointCloud


def get_random_push(
    n_params: int,
    obj_states: np.ndarray,
    obj_shape: tuple[float, float, float],
    tool_offset: np.ndarray = np.array([0, 0, 0, 1, 0, 0, 0]),
    rotation_range: tuple[float, float] = (0, 2 * np.pi),  # ang to push from
    side_range: tuple[float, float] = (-0.4, 0.4),  # relative side offset
    distance_range: tuple[float, float] = (0, 0.3),  # push distance
    pre_push_offset: float = 0.02,  # pre-push offset at the beginning
    duration: float = 2,  # total time to complete the push
    dt: float = 0.1,  # time step of the path
    max_speed: float = 0.5,  # assume it will never exceed this speed
    max_acc: float = 1,  # assume it will never exceed this acceleration
):
    """Get a random push parameter and the corresponding path"""
    push_params = generate_push_params(
        n_params, rotation_range, side_range, distance_range
    )
    times, ws_paths = generate_path_from_params(
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


def generate_push_params(
    n_params: int,
    # Angle to push from (will be discretized to pi / 2)
    rotation_range: tuple[float, float] = (0, 2 * np.pi),
    # Relative push side offset
    side_range: tuple[float, float] = (-0.4, 0.4),
    # Push distance
    distance_range: tuple[float, float] = (0, 0.3),
):
    """Generate a random push parameter"""
    # Push rotation
    # (discrete over pi / 2, normalized by 2 * pi) -> (0, 0.25, 0.5, 0.75...)
    rotations = np.random.uniform(*rotation_range, n_params)
    rotations = (rotations / (np.pi / 2)).astype(int) / 4
    # Push side offset (relative value w.r.t. the object size)
    sides = np.random.uniform(*side_range, n_params)
    # Push distance (absolute value)
    distances = np.random.uniform(*distance_range, n_params)

    # Stack all parameters (n_params, 3)
    push_params = np.stack([rotations, sides, distances], axis=-1)
    return push_params


def generate_path_from_params(
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
    """Generate a workspace path from the push parameters"""
    # Unpack parameters
    assert push_params.ndim == 2 and push_params.shape[1] == 3
    assert obj_states.shape[0] == push_params.shape[0]
    n_data = push_params.shape[0]
    rotations, sides, distances = push_params.T

    # Convert the normalized rotation to the absolute value
    push_sides = np.round(rotations * 4)
    rotations = push_sides * (np.pi / 2)
    # Covert the relative side offset to the absolute value
    # the object is approximated as an AABB
    w, l, h = obj_shape
    mask_odd = push_sides % 2 == 1
    sizes = np.where(mask_odd, l, w)
    sides = np.where(mask_odd, w * sides, l * sides)

    # Get local path (x, y) w.r.t. the object
    # direction vectors (N, 2)
    dir_vecs = np.stack([np.cos(rotations), np.sin(rotations)], axis=1)
    # side offset vectors (N, 2)
    side_vecs = np.stack([-dir_vecs[:, 1], dir_vecs[:, 0]], axis=1)
    # add pre-push offset to avoid hitting object at the beginning
    distances = distances + pre_push_offset  # (N, 1)
    # start point (N, 2)
    starts = (
        dir_vecs * (sizes / 2 + pre_push_offset)[:, None]
        + sides[:, None] * side_vecs
    )

    # Check constraints before generating path
    peak_speed, peak_acc = get_sin_velocity_profile_peak(distances, duration)
    assert np.all(peak_speed <= max_speed) and np.all(peak_acc <= max_acc)
    # Generate the waypoint distance from start
    n_steps = int(duration / dt)
    t_paths = np.tile(np.linspace(0, duration, n_steps), (n_data, 1))  # (N, T)
    distances = distances[:, None]  # (N, 1)
    dists = sin_velocity_profile(t_paths, distances, duration)  # (N, T)

    # Generate path in local object frame
    # Position part
    # local = start - dist * dir_vec, (N, T, 2)
    local_xy = starts[:, None, :] - dists[:, :, None] * dir_vecs[:, None, :]
    # combine height (N, T, 3)
    local_z = np.zeros((n_data, n_steps, 1))
    local_pos = np.concatenate([local_xy, local_z], axis=2)

    # Rotation part
    # Batch operation for the following operations
    # for:
    #     pose_delta = rotate_z @ reflect_z @ tool_offset
    #     global_pose = obj_pose @ Pose(local_pos[i]) @ pose_delta

    # Transform matrix will be size (n_data, n_steps, 4, 4)
    # add reflection and offset - ensure ee pointing down
    t_rotate_z = np.tile(np.eye(4)[None, None, :, :], (n_data, 1, 1, 1))
    t_rotate_z[:, 0, :3, :3] = euler_to_matrix("z", rotations + np.pi)
    t_reflect_z = np.eye(4)[None, None, :, :]
    t_reflect_z[0, 0, :3, :3] = euler_to_matrix("x", np.pi)
    t_tool_offset = flat_to_matrix(tool_offset)[None, None, :, :]
    t_delta = t_rotate_z @ t_reflect_z @ t_tool_offset

    # Combine as workspace path
    t_local_pos = np.tile(np.eye(4)[None, None, :, :], (n_data, n_steps, 1, 1))
    t_local_pos[:, :, :3, 3] = local_pos
    # get local poses
    t_local = t_local_pos @ t_delta
    # convert to world frame given the object world pose
    t_obj = flat_to_matrix(obj_states)
    t_global = t_obj[:, None, :, :] @ t_local

    # Flatten to 7D
    ws_pos = t_global[:, :, :3, 3]
    ws_quat = matrix_to_quat(t_global[:, :, :3, :3].reshape(-1, 3, 3))
    ws_quat = ws_quat.reshape(n_data, n_steps, 4)
    ws_paths = np.concatenate([ws_pos, ws_quat], axis=-1)
    return t_paths, ws_paths


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
    
    # Reshape angles to [N_points, 1] for concatenation
    angles_reshaped = angles.reshape(-1, 1)
    
    # Concatenate: points(3) + normals(3) + angles(1) = 7 features
    push_params = np.concatenate([points, normals, angles_reshaped], axis=1)    # [N_points, 7]

    
    # Generate paths
    times, ws_paths = generate_circular_push_path(
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


def get_random_circular_push(
    n_params: int,
    obj_states: np.ndarray,
    obj_shape: tuple[float, float, float],
    tool_offset: np.ndarray = np.array([0, 0, 0, 1, 0, 0, 0]),
    rotation_range: tuple[float, float] = (0, 2 * np.pi),  # ang to push from
    pre_push_offset: float = 0.02,  # pre-push offset at the beginning
    duration: float = 2,  # total time to complete the push
    dt: float = 0.1,  # time step of the path
    max_speed: float = 0.5,  # assume it will never exceed this speed
    max_acc: float = 1,  # assume it will never exceed this acceleration
):
    """Get a random push parameter and the corresponding path"""
    push_params = generate_circular_push_params(
        n_params, rotation_range
    )
    times, ws_paths = generate_circular_push_path(
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


def generate_circular_push_params(
    n_params: int,
    # Angle to push from (0 to 2pi)
    angle_range: tuple[float, float] = (0, 2 * np.pi),
):
    """Generate random push angles (N, 1)"""
    # Push angles
    angles = np.random.uniform(*angle_range, n_params)
    angles = angles % (2 * np.pi) # Wrap to [0, 2Pi]

    # Shape becomes (N, 1)
    push_params = np.stack([angles], axis=-1)
    return push_params


def generate_circular_push_path(
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
    # push_params shape: [N, 7] where columns are [x, y, z, nx, ny, nz, angle]
    assert push_params.ndim == 2 and push_params.shape[1] == 7, f"Expected push_params shape (N, 7), got {push_params.shape}"
    
    n_data = push_params.shape[0]
    
    # Unpack data
    contact_points = push_params[:, :3]      # [N, 3] - (x, y, z)
    contact_normals = push_params[:, 3:6]    # [N, 3] - (nx, ny, nz)
    angles = push_params[:, 6]               # [N,] - angle in radians
    
    # Calculate Object "Radius" (Max extent from center)
    length_x, length_y, _ = obj_shape
    # Using half-diagonal ensures we clear the corners of a box
    obj_max_r = np.sqrt(length_x**2 + length_y**2) / 2.0

    # 1. Define Direction Vectors (Point to center)
    push_dir_x = -np.cos(angles)
    push_dir_y = -np.sin(angles)
    dir_vecs = np.stack([push_dir_x, push_dir_y], axis=1) # (N, 2)

    # 2. Define Start Points
    # Start distance = Max object extent + Clearance
    start_dist = obj_max_r + pre_push_offset
    starts = np.stack([np.cos(angles), np.sin(angles)], axis=1) * start_dist

    # 3. Velocity Profile
    # Total distance is diameter of the effective circle (Start -> Center -> End)
    # --- FIX 2: Create an array of distances for broadcasting ---
    dist_scalar = start_dist * 2
    total_push_dists = np.full((n_data, 1), dist_scalar)

    peak_speed, peak_acc = get_sin_velocity_profile_peak(total_push_dists, duration)
    assert np.all(peak_speed <= max_speed) and np.all(peak_acc <= max_acc)
    
    n_steps = int(duration / dt)
    t_paths = np.tile(np.linspace(0, duration, n_steps), (n_data, 1))
    
    # Now this works because total_push_dists is (N, 1)
    path_dists = sin_velocity_profile(t_paths, total_push_dists, duration)

    # 4. Generate Local Path
    # Position: Start + (distance_traveled * direction_vector)
    # starts shape: (N, 1, 2) | path_dists shape: (N, T, 1) | dir_vecs shape: (N, 1, 2)
    local_xy = starts[:, None, :] + path_dists[:, :, None] * dir_vecs[:, None, :]
    local_z = np.zeros((n_data, n_steps, 1))
    local_pos = np.concatenate([local_xy, local_z], axis=2)

    # 5. Rotation Logic (EE orientation)
    # Tool faces the center (angle + PI)
    t_rotate_z = np.tile(np.eye(4)[None, None, :, :], (n_data, 1, 1, 1))
    
    # Vectorized rotation matrix construction is faster, but loop is fine for clarity
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