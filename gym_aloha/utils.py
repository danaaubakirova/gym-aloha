import numpy as np

def sample_box_pose(seed=None):
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    rng = np.random.RandomState(seed)

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose(seed=None):
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    rng = np.random.RandomState(seed)

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

def sample_insertion_and_transfer_pose(seed=None):
    """Samples random positions for peg, socket, and cube for the insertion and transfer tasks."""
    rng = np.random.RandomState(seed)

    # Sample peg position
    peg_x_range = [0.1, 0.2]
    peg_y_range = [0.4, 0.6]
    peg_z_range = [0.05, 0.05]
    peg_ranges = np.vstack([peg_x_range, peg_y_range, peg_z_range])
    peg_position = rng.uniform(peg_ranges[:, 0], peg_ranges[:, 1])
    peg_quat = np.array([1, 0, 0, 0])  # Fixed orientation
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Sample socket position
    socket_x_range = [-0.2, -0.1]
    socket_y_range = [0.4, 0.6]
    socket_z_range = [0.05, 0.05]
    socket_ranges = np.vstack([socket_x_range, socket_y_range, socket_z_range])
    socket_position = rng.uniform(socket_ranges[:, 0], socket_ranges[:, 1])
    socket_quat = np.array([1, 0, 0, 0])  # Fixed orientation
    socket_pose = np.concatenate([socket_position, socket_quat])

    # Sample cube position (transfer task) - move it closer to peg and socket
    cube_x_range = [-0.15, 0.15]  # Cube is now closer to the peg and socket
    cube_y_range = [0.6, 0.7]  # Slightly overlapping in y direction but still separated
    cube_z_range = [0.05, 0.05]  # Cube height remains constant
    cube_ranges = np.vstack([cube_x_range, cube_y_range, cube_z_range])
    cube_position = rng.uniform(cube_ranges[:, 0], cube_ranges[:, 1])
    cube_quat = np.array([1, 0, 0, 0])  # Fixed orientation
    cube_pose = np.concatenate([cube_position, cube_quat])

    return peg_pose, socket_pose, cube_pose
