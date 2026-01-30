import numpy as np
import open3d as o3d

class ObjectPointCloud:
    def __init__(
        self,
        mesh_path,
        z_height=0.0,
        num_points=1000,
        thickness=0.001,
        center_pos=np.array([0, 0, 0]),
        rpy=np.array([0, 0, 0]),
        sort_by_angle=True,
    ):
        # 1. Load and move mesh origin to center
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.mesh.compute_vertex_normals()
        mesh_center = self.mesh.get_center()
        self.mesh.translate(-mesh_center)

        # 2. Dense initial sampling
        initial_sample_size = num_points * 100
        initial_pcd = self.mesh.sample_points_uniformly(number_of_points=initial_sample_size)
        points = np.asarray(initial_pcd.points)
        normals = np.asarray(initial_pcd.normals)

        # 3. Filter points to the slice
        target_z = z_height
        z_mask = np.abs(points[:, 2] - target_z) <= thickness
        sliced_points = points[z_mask]
        sliced_normals = normals[z_mask]

        # 4. Uniformly resample from the slice
        if len(sliced_points) >= num_points:
            idx = np.random.choice(len(sliced_points), num_points, replace=False)
        else:
            print(f"Warning: Only {len(sliced_points)} points in slice. "
                  f"Resampling with replacement to reach {num_points}.")
            idx = np.random.choice(len(sliced_points), num_points, replace=True)
        
        points = sliced_points[idx]
        normals = sliced_normals[idx]

        # 5. Sort by angle
        if sort_by_angle:
            angles = np.arctan2(points[:, 1], points[:, 0])

            sorted_indices = np.argsort(angles)

            self.points = points[sorted_indices]
            self.normals = normals[sorted_indices]
            self.angles = angles[sorted_indices]
            
        # 6. Transform to world coordinates
        T = self.rpy_to_matrix(center_pos, rpy)
        self.points, self.normals = self.transform(T)

    def transform(self, T):
        rotation_matrix = T[:3, :3]
        points_h = np.hstack([self.points, np.ones((self.points.shape[0], 1))])
        transformed_points = (T @ points_h.T).T[:, :3]
        transformed_normals = (rotation_matrix @ self.normals.T).T
        return transformed_points, transformed_normals

    def rpy_to_matrix(self, translation, rpy):
        R = o3d.geometry.get_rotation_matrix_from_xyz(rpy)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = translation
        return T

    def show(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.normals = o3d.utility.Vector3dVector(self.normals)

        # Check size
        aabb = self.mesh.get_axis_aligned_bounding_box()
        aabb.color = (0, 1, 0)
        dx, dy, dz = aabb.get_extent()
        print(f"\nOriginal Mesh AABB extent: {dx:.4f}, {dy:.4f}, {dz:.4f}")

        pcd_aabb = pcd.get_axis_aligned_bounding_box()
        pcd_aabb.color = (1, 0, 0)
        dx, dy, dz = pcd_aabb.get_extent()
        print(f"Point Cloud AABB extent: {dx:.4f}, {dy:.4f}, {dz:.4f}")
        print(f"Number of points: {len(self.points)}")

        o3d.visualization.draw_geometries([pcd, pcd_aabb, aabb])


if __name__ == "__main__":
    try:
        pcd = ObjectPointCloud(
            # "assets/cracker_box_flipped/textured.obj",
            "assets/mustard_bottle_flipped/textured.obj",
            z_height=0.0,
            num_points=1000,
            thickness=0.001
        )
        pcd.show()
    except ValueError as e:
        print(f"\n Error: {e}")