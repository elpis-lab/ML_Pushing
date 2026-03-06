import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from object_cloud_point import ObjectPointCloud


NORMALIZATION = True
WITH_LLC      = True
BATCH_SIZE    = 64
LR            = 1e-3
EPOCHS        = 100
WINDOWSIZE    = 10
# INPUT_DIM     = 7
OUTPUT_DIM    = 3
NOVEL_OBJ     = "mustard_bottle_flipped"


def get_local_curvature(points, window_size):
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

# ---------------------------------------------------------------------------
# 1.  Dataset
# ---------------------------------------------------------------------------
class LoadDataset(Dataset):
    def __init__(self, normalization=True, with_llc=True):
        x1 = np.load("data/x_cracker_box_flipped_100.npy")
        x2 = np.load("data/x_mustard_bottle_flipped_100.npy")
        x3 = np.load("data/x_banana_flipped_100.npy")
        x4 = np.load("data/x_box_1_100.npy")
        x5 = np.load("data/x_circle_1_100.npy")
        # x6 = np.load("data/x_half_1_1000.npy")
        # x7 = np.load("data/x_quater_1_1000.npy")

        y1 = np.load("data/y_cracker_box_flipped_100.npy")
        y2 = np.load("data/y_mustard_bottle_flipped_100.npy")
        y3 = np.load("data/y_banana_flipped_100.npy")
        y4 = np.load("data/y_box_1_100.npy")
        y5 = np.load("data/y_circle_1_100.npy")
        # y6 = np.load("data/y_half_1_1000.npy")
        # y7 = np.load("data/y_quater_1_1000.npy")

        # [px, py, pz, nx, ny, nv, angle]
        self.x_data = np.concatenate([x1, x3, x4, x5], axis=0).astype(np.float32)
        # [dx, dy, dtheta]
        self.y_data = np.concatenate([y1, y3, y4, y5], axis=0).astype(np.float32)

        # TEST: Hold out mustard for evaluation
        self.x_test = x2.astype(np.float32)
        self.y_test = y2.astype(np.float32)

        if with_llc:
            # Add curvature to training data
            points = self.x_data[:, :3]
            curvatures = get_local_curvature(points, WINDOWSIZE)
            curvatures_reshaped = curvatures.reshape(-1, 1)
            # [px, py, pz, nx, ny, nv, angle, curvature]
            self.x_data = np.concatenate([self.x_data, curvatures_reshaped], axis=1)
            
            # Add curvature to test data
            points_test = self.x_test[:, :3]
            curvatures_test = get_local_curvature(points_test, WINDOWSIZE)
            curvatures_test_reshaped = curvatures_test.reshape(-1, 1)
            self.x_test = np.concatenate([self.x_test, curvatures_test_reshaped], axis=1)
            
            print(f"Dataset with curvature: input dimension = {self.x_data.shape[1]}")
        else:
            print(f"Dataset without curvature: input dimension = {self.x_data.shape[1]}")

        self.num_samples = self.x_data.shape[0]
        self.num_test_samples = self.x_test.shape[0]
        self.input_dim = self.x_data.shape[1]
        
        assert self.x_data.shape[0] == self.y_data.shape[0], \
            "x and y datasets must have the same number of samples"
        assert self.x_test.shape[0] == self.y_test.shape[0], \
            "x_test and y_test must have the same number of samples"
        
        # Normalization
        # Store mean/std so the same transform can be applied at inference time.
        if normalization:
            # Compute statistics from TRAINING data only
            self.x_mean = self.x_data.mean(axis=0)
            self.x_std  = self.x_data.std(axis=0) + 1e-8
            self.y_mean = self.y_data.mean(axis=0)
            self.y_std  = self.y_data.std(axis=0) + 1e-8

            # Normalize training data
            self.x_data = (self.x_data - self.x_mean) / self.x_std
            self.y_data = (self.y_data - self.y_mean) / self.y_std
            
            # Normalize test data with TRAINING statistics
            self.x_test = (self.x_test - self.x_mean) / self.x_std
            self.y_test = (self.y_test - self.y_mean) / self.y_std
        else:
            # Identity transform
            self.x_mean = np.zeros(self.x_data.shape[1], dtype=np.float32)
            self.x_std  = np.ones(self.x_data.shape[1], dtype=np.float32)
            self.y_mean = np.zeros(self.y_data.shape[1], dtype=np.float32)
            self.y_std  = np.ones(self.y_data.shape[1], dtype=np.float32)

    # PyTorch Dataset interface
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.tensor(self.x_data[idx])
        y = torch.tensor(self.y_data[idx])
        return x, y


# ---------------------------------------------------------------------------
# 2.  Model
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim=8, output_dim=3, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
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
        self.loss_fn   = nn.MSELoss()

        # ── For recording curves ──
        self.train_losses = []
        self.val_losses = []

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

            # ── Record losses ──
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # if (epoch + 1) % 10 == 0:
                # print(f"Epoch [{epoch+1:4d}/{self.epochs}]  "
                    #   f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

            # ── save best model ──
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pth")

        print(f"Training complete. Best val loss: {best_val_loss:.6f}")
        
        # ── Save loss curves ──
        self.save_curves()
        
        return self.model

    def save_curves(self):
        """Save training and validation curves as numpy files and plot"""
        plt.figure(figsize=(10, 6))
        epochs = np.arange(1, len(self.train_losses) + 1)
        
        plt.plot(epochs, self.train_losses, label='Training Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title('Training and Validation Loss Curves', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/loss_curves.png', dpi=300)
        print("Saved loss curves plot")
        plt.close()


# ---------------------------------------------------------------------------
# 4.  Test Set Evaluation
# ---------------------------------------------------------------------------
class TestEvaluator:
    def __init__(self, model, device, test_data, norm_stats):
        """
        model      : trained MLP
        device     : torch device
        test_data  : tuple of (x_test, y_test) - already normalized
        norm_stats : dict with normalization statistics
        """
        self.model = model.to(device).eval()
        self.device = device
        self.x_test, self.y_test = test_data
        self.y_mean = norm_stats['y_mean']
        self.y_std = norm_stats['y_std']
    
    def evaluate(self):
        """
        Evaluate model on held-out test set and return metrics
        """
        x_tensor = torch.tensor(self.x_test).to(self.device)
        y_tensor = torch.tensor(self.y_test).to(self.device)
        
        with torch.no_grad():
            y_pred_norm = self.model(x_tensor)
        
        # Compute loss on normalized predictions
        loss_fn = nn.MSELoss()
        test_loss_normalized = loss_fn(y_pred_norm, y_tensor).item()
        
        # Denormalize for real-world metrics
        y_pred = y_pred_norm.cpu().numpy() * self.y_std + self.y_mean
        y_true = y_tensor.cpu().numpy() * self.y_std + self.y_mean
        
        # Compute metrics
        mse = np.mean((y_pred - y_true) ** 2)
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(mse)
        
        # Per-output metrics
        mse_per_output = np.mean((y_pred - y_true) ** 2, axis=0)
        mae_per_output = np.mean(np.abs(y_pred - y_true), axis=0)
        
        print("\n" + "="*60)
        print("TEST SET EVALUATION")
        print("="*60)
        print(f"Test samples: {len(y_true)}")
        print(f"  To_MSE:  {mse:.6f}")
        print(f"  dx_MSE: {mse_per_output[0]:.6f}")
        print(f"  dy_MSE: {mse_per_output[1]:.6f}")
        print(f"  dθ_MSE: {mse_per_output[2]:.6f}")
        print("="*60 + "\n")
        
        # Save detailed results
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mse_per_output': mse_per_output,
            'mae_per_output': mae_per_output,
            'predictions': y_pred,
            'ground_truth': y_true
        }
        
        return results


# ---------------------------------------------------------------------------
# 5.  Prediction on a novel object (for visualization/deployment)
# ---------------------------------------------------------------------------
class Predicting:
    def __init__(self, model, device, norm_stats, with_llc=True):
        self.model  = model.to(device).eval()
        self.device = device
        self.x_mean = norm_stats['x_mean']
        self.x_std  = norm_stats['x_std']
        self.y_mean = norm_stats['y_mean']
        self.y_std  = norm_stats['y_std']
        self.with_llc = with_llc

    @staticmethod
    def mesh_to_pcd(obj_name, num_points=1000):
        mesh_path = f"assets/{obj_name}/textured.obj"
        pcd = ObjectPointCloud(
            mesh_path,
            z_height=0.0,
            num_points=num_points,
            thickness=0.001,
            sort_by_angle=True,
        )
        return pcd.points, pcd.normals, pcd.angles

    def build_input_features(self, obj_name, num_points=1000):
        points, normals, angles = self.mesh_to_pcd(obj_name, num_points)
        
        if self.with_llc:
            curvatures = get_local_curvature(points, WINDOWSIZE)
            features = np.column_stack([
                points,                             # px, py, pz  (N,3)
                normals,                            # nx, ny, nz  (N,3)
                angles.reshape(-1, 1),              # pushing_angle (N,1)
                curvatures.reshape(-1, 1),          # curvature     (N,1)
            ]).astype(np.float32)                   # (N, 8)
        else:
            features = np.column_stack([
                points,                             # px, py, pz  (N,3)
                normals,                            # nx, ny, nz  (N,3)
                angles.reshape(-1, 1),              # pushing_angle (N,1)
            ]).astype(np.float32)                   # (N, 7)

        return features, points

    def predict(self, obj_name, num_points=1000, output_file="prediction.npy"):
        """
        Full prediction pipeline:
            mesh → point cloud → features → normalize → model → denormalize → save
        Output .npy shape: (N, 4)  —  [px, py, dx, dy]
        """
        features, points = self.build_input_features(obj_name, num_points)
        
        print(f"Prediction input features shape: {features.shape}")

        # Normalize with training statistics
        features_norm = (features - self.x_mean) / self.x_std

        # Forward pass
        x_tensor = torch.tensor(features_norm).to(self.device)   # (N, 7 or 8)
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
# 6.  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── 1. Load dataset (mustard held out for testing) ──
    dataset = LoadDataset(normalization=NORMALIZATION, with_llc=WITH_LLC)
    print(f"Loaded {dataset.num_samples} training samples")
    print(f"Held out {dataset.num_test_samples} test samples (mustard)")
    
    # Get actual input dimension from dataset
    INPUT_DIM = dataset.input_dim

    # ── 2. Build model ──
    model = MLP(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
    # print(model)

    # ── 3. Train ──
    trainer = Training(model, dataset, device,
                       batch_size=BATCH_SIZE, lr=LR, epochs=EPOCHS)
    model = trainer.train()

    # ── 4. Save normalization stats ──
    norm_stats = {
        'x_mean': dataset.x_mean,
        'x_std':  dataset.x_std,
        'y_mean': dataset.y_mean,
        'y_std':  dataset.y_std,
    }
    # np.savez("norm_stats.npz", **norm_stats)

    # ── 5. Evaluate on held-out TEST SET (mustard ground truth) ──
    model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
    
    test_data = (dataset.x_test, dataset.y_test)
    evaluator = TestEvaluator(model, device, test_data, norm_stats)
    test_results = evaluator.evaluate()
    
    # Save test results
    # np.savez("data/test_results.npz", **test_results)
    # print("Saved test results to data/test_results.npz")

    # ── 6. Generate predictions for visualization (optional) ──
    predictor = Predicting(model, device, norm_stats, with_llc=WITH_LLC)
    predictor.predict(NOVEL_OBJ, num_points=1000, output_file="data/prediction.npy")