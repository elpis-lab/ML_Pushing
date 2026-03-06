import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from object_cloud_point import ObjectPointCloud


# ---------------------------------------------------------------------------
# 1.  Dataset
# ---------------------------------------------------------------------------
class LoadDataset(Dataset):
    # def __init__(self, x_file, y_file, normalization=True):
    def __init__(self, normalization=True):
        # self.x_data = np.load(x_file).astype(np.float32)  # (N, 8)
        # self.y_data = np.load(y_file).astype(np.float32)  # (N, 3)
        x1 = np.load("data/x_cracker_box_flipped_1000.npy")
        x2 = np.load("data/x_mustard_bottle_flipped_1000.npy")
        x3 = np.load("data/x_banana_flipped_1000.npy")
        x4 = np.load("data/x_box_1_1000.npy")
        x5 = np.load("data/x_circle_1_1000.npy")
        x6 = np.load("data/x_half_1_1000.npy")
        x7 = np.load("data/x_quater_1_1000.npy")

        y1 = np.load("data/y_cracker_box_flipped_1000.npy")
        y2 = np.load("data/y_mustard_bottle_flipped_1000.npy")
        y3 = np.load("data/y_banana_flipped_1000.npy")
        y4 = np.load("data/y_box_1_1000.npy")
        y5 = np.load("data/y_circle_1_1000.npy")
        y6 = np.load("data/y_half_1_1000.npy")
        y7 = np.load("data/y_quater_1_1000.npy")

        self.x_data = np.concatenate([x1, x2, x3, x5], axis=0).astype(np.float32)
        self.y_data = np.concatenate([y1, y2, y3, y5], axis=0).astype(np.float32)
        

        assert self.x_data.shape[0] == self.y_data.shape[0], \
            "x and y datasets must have the same number of samples"

        self.num_samples = self.x_data.shape[0]

        self.push_points   = self.x_data[:, :2]   # px, py
        self.push_normals  = self.x_data[:, 3:5]  # nx, ny
        self.pushing_angles = self.x_data[:, 6]   # pushing_angle
        self.curvatures    = self.x_data[:, 7]    # curvature

        self.dx      = self.y_data[:, 0]
        self.dy      = self.y_data[:, 1]
        self.dtheta  = self.y_data[:, 2]

        # ── normalization ──
        # Store mean/std so the same transform can be applied at inference time.
        if normalization:
            self.x_mean = self.x_data.mean(axis=0)            # (8,)
            self.x_std  = self.x_data.std(axis=0) + 1e-8      # avoid div-by-zero
            self.y_mean = self.y_data.mean(axis=0)            # (3,)
            self.y_std  = self.y_data.std(axis=0) + 1e-8

            self.x_data = (self.x_data - self.x_mean) / self.x_std
            self.y_data = (self.y_data - self.y_mean) / self.y_std
        else:
            # Identity transform — still store so prediction code is uniform
            self.x_mean = np.zeros(self.x_data.shape[1], dtype=np.float32)
            self.x_std  = np.ones(self.x_data.shape[1], dtype=np.float32)
            self.y_mean = np.zeros(self.y_data.shape[1], dtype=np.float32)
            self.y_std  = np.ones(self.y_data.shape[1], dtype=np.float32)

    # ── PyTorch Dataset interface ──
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.tensor(self.x_data[idx])   # (8,)
        y = torch.tensor(self.y_data[idx])   # (3,)
        return x, y


# ---------------------------------------------------------------------------
# 2.  Model
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim=8, output_dim=3, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]   # default architecture

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))  # no activation — raw regression output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 3.  Training
# ---------------------------------------------------------------------------
class Training:
    def __init__(self, model, dataset, device,
                 batch_size=64, lr=1e-3, epochs=100, val_ratio=0.1):
        self.model  = model.to(device)
        self.device = device
        self.epochs = epochs

        # ── train / val split ──
        n = len(dataset)
        n_val = int(n * val_ratio)
        train_set, val_set = torch.utils.data.random_split(dataset, [n - n_val, n_val])

        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()   # regression → MSE, not cross-entropy

    def train(self):
        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            # ── training ──
            self.model.train()
            train_loss = 0.0
            for x_batch, y_batch in self.train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                pred = self.model(x_batch)
                loss = self.loss_fn(pred, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * x_batch.size(0)
            train_loss /= len(self.train_loader.dataset)

            # ── validation ──
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in self.val_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    pred    = self.model(x_batch)
                    val_loss += self.loss_fn(pred, y_batch).item() * x_batch.size(0)
            val_loss /= len(self.val_loader.dataset)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1:4d}/{self.epochs}]  "
                      f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

            # ── save best model ──
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pth")

        print(f"Training complete. Best val loss: {best_val_loss:.6f}")
        return self.model


# ---------------------------------------------------------------------------
# 4.  Prediction on a novel object
# ---------------------------------------------------------------------------
class Predicting:
    def __init__(self, model, device, norm_stats):
        """
        model      : trained MLP (weights already loaded)
        device     : torch device
        norm_stats : dict with keys 'x_mean','x_std','y_mean','y_std' (numpy arrays)
        """
        self.model  = model.to(device).eval()
        self.device = device
        self.x_mean = norm_stats['x_mean']
        self.x_std  = norm_stats['x_std']
        self.y_mean = norm_stats['y_mean']
        self.y_std  = norm_stats['y_std']

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def mesh_to_pcd(obj_name, num_points=1000):
        """Convert an OBJ mesh to a point cloud."""
        mesh_path = f"assets/{obj_name}/textured.obj"
        pcd = ObjectPointCloud(
            mesh_path,
            z_height=0.0,
            num_points=num_points,
            thickness=0.001,
            sort_by_angle=True,
        )
        return pcd.points, pcd.normals, pcd.angles   # (N,3), (N,3), (N,)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def get_local_curvature(points, window_size=11):
        """
        Estimate local 2D curvature at each point via least-squares circle fit.
        points: (N, 3)  — only x,y are used.
        """
        num_points  = len(points)
        curvatures  = np.zeros(num_points, dtype=np.float32)
        points_xy   = points[:, :2]
        half_win    = window_size // 2

        for i in range(num_points):
            indices    = np.arange(i - half_win, i + half_win + 1) % num_points
            neighborhood = points_xy[indices]

            x = neighborhood[:, 0]
            y = neighborhood[:, 1]

            # Least-squares circle: x² + y² = 2ax + 2by + C
            A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
            B = x**2 + y**2

            try:
                res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
                a, b, C = res

                R_squared = C + a**2 + b**2
                if R_squared <= 0:
                    curvatures[i] = 0.0
                else:
                    R = np.sqrt(R_squared)
                    curvatures[i] = 1.0 / R if R > 1e-6 else 0.0
            except np.linalg.LinAlgError:
                curvatures[i] = 0.0

        return curvatures

    # ──────────────────────────────────────────────────────────────────────
    def build_input_features(self, obj_name, num_points=1000):
        """
        Given a novel object name, build the (N, 8) feature matrix
        that matches the training input format:
            [px, py, pz, nx, ny, nz, pushing_angle, curvature]
        """
        points, normals, angles = self.mesh_to_pcd(obj_name, num_points)
        curvatures = self.get_local_curvature(points)

        # angles from ObjectPointCloud is (N,) — reshape to column
        features = np.column_stack([
            points,                             # px, py, pz  (N,3)
            normals,                            # nx, ny, nz  (N,3)
            angles.reshape(-1, 1),              # pushing_angle (N,1)
            curvatures.reshape(-1, 1),          # curvature     (N,1)
        ]).astype(np.float32)                   # (N, 8)

        return features, points

    # ──────────────────────────────────────────────────────────────────────
    def predict(self, obj_name, num_points=1000, output_file="prediction.npy"):
        """
        Full prediction pipeline:
            mesh → point cloud → features → normalize → model → denormalize → save
        Output .npy shape: (N, 4)  —  [px, py, dx, dy]
        """
        features, points = self.build_input_features(obj_name, num_points)

        # Normalize with training statistics
        features_norm = (features - self.x_mean) / self.x_std

        # Forward pass
        x_tensor = torch.tensor(features_norm).to(self.device)   # (N, 8)
        with torch.no_grad():
            y_pred_norm = self.model(x_tensor)                    # (N, 3)

        # Denormalize predictions back to original scale
        y_pred = y_pred_norm.cpu().numpy() * self.y_std + self.y_mean  # (N, 3)

        # Extract dx, dy (columns 0 and 1)
        px = points[:, 0].reshape(-1, 1)   # (N, 1)
        py = points[:, 1].reshape(-1, 1)   # (N, 1)
        dx = y_pred[:, 0].reshape(-1, 1)   # (N, 1)
        dy = y_pred[:, 1].reshape(-1, 1)   # (N, 1)

        output = np.hstack([px, py, dx, dy])   # (N, 4)
        np.save(output_file, output)
        print(f"Saved predictions to {output_file}  shape={output.shape}")
        return output


# ---------------------------------------------------------------------------
# 5.  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ── config ──
    X_FILE        = "data/x_cracker_box_flipped_1000.npy"
    Y_FILE        = "data/y_cracker_box_flipped_1000.npy"
    NORMALIZATION = True
    BATCH_SIZE    = 64
    LR           = 1e-3
    EPOCHS       = 100
    INPUT_DIM    = 8
    OUTPUT_DIM   = 3
    NOVEL_OBJ    = "quater_1"   # name of the novel object to predict on

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── 1. Load dataset ──
    # dataset = LoadDataset(X_FILE, Y_FILE, normalization=NORMALIZATION)
    dataset = LoadDataset(normalization=NORMALIZATION)
    print(f"Loaded {dataset.num_samples} samples")

    # ── 2. Build model ──
    model = MLP(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
    print(model)

    # ── 3. Train ──
    trainer = Training(model, dataset, device,
                       batch_size=BATCH_SIZE, lr=LR, epochs=EPOCHS)
    model = trainer.train()

    # ── 4. Save normalization stats alongside the model ──
    norm_stats = {
        'x_mean': dataset.x_mean,
        'x_std':  dataset.x_std,
        'y_mean': dataset.y_mean,
        'y_std':  dataset.y_std,
    }
    np.savez("norm_stats.npz", **norm_stats)

    # ── 5. Predict on novel object ──
    # Load best model weights
    # model.load_state_dict(torch.load("best_model.pth", map_location=device))

    model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))


    predictor = Predicting(model, device, norm_stats)
    predictor.predict(NOVEL_OBJ, num_points=1000, output_file="data/prediction.npy")