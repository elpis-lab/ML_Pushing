import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt

from object_cloud_point import ObjectPointCloud


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
NORMALIZATION    = True
WITH_LLC         = False
BATCH_SIZE       = 32          # samples per task per episode
LR_INNER         = 1e-2        # α  – fast adaptation learning rate
LR_OUTER         = 1e-3        # β  – meta learning rate
EPOCHS           = 200         # meta-training epochs
INNER_STEPS      = 5           # gradient steps in the inner loop
K_SHOT           = 16          # support samples per task (K-shot)
QUERY_SIZE       = 32          # query samples per task
WINDOWSIZE       = 10
OUTPUT_DIM       = 3
NOVEL_OBJ        = "mustard_bottle_flipped"
FINETUNE_STEPS   = 20          # inner steps during test-time fine-tuning
FINETUNE_LR      = 1e-2



# ---------------------------------------------------------------------------
# 1.  Task dataset – one tensor per object
# ---------------------------------------------------------------------------
class TaskCollection:
    """
    Loads all objects and exposes them as a list of (X, Y) tensor tasks.
    Test task (mustard bottle) is held out for meta-test evaluation.
    """

    TRAIN_OBJECTS = [
        ("cracker_box_flipped", "data/x_cracker_box_flipped_1000.npy",  "data/y_cracker_box_flipped_1000.npy"),
        ("banana_flipped",      "data/x_banana_flipped_1000.npy",       "data/y_banana_flipped_1000.npy"),
        ("box_1",               "data/x_box_1_1000.npy",                "data/y_box_1_1000.npy"),
        ("circle_1",            "data/x_circle_1_1000.npy",             "data/y_circle_1_1000.npy"),
    ]
    TEST_OBJECT = (
        "mustard_bottle_flipped",
        "data/x_mustard_bottle_flipped_1000.npy",
        "data/y_mustard_bottle_flipped_1000.npy",
    )

    def __init__(self, normalization=True, with_llc=False):
        self.with_llc      = with_llc
        self.normalization = normalization

        # --- build raw arrays for every training object ---
        train_xs, train_ys = [], []
        for name, xp, yp in self.TRAIN_OBJECTS:
            x_raw = np.load(xp).astype(np.float32)
            y_raw = np.load(yp).astype(np.float32)
            x_raw = self._append_curvature(x_raw)
            train_xs.append(select_features(x_raw, with_llc))
            train_ys.append(y_raw)

        # concatenate ALL training data for normalization statistics
        x_all = np.concatenate(train_xs, axis=0)
        y_all = np.concatenate(train_ys, axis=0)

        if normalization:
            self.x_mean = x_all.mean(0)
            self.x_std  = x_all.std(0) + 1e-8
            self.y_mean = y_all.mean(0)
            self.y_std  = y_all.std(0) + 1e-8
        else:
            self.x_mean = np.zeros(x_all.shape[1], dtype=np.float32)
            self.x_std  = np.ones(x_all.shape[1],  dtype=np.float32)
            self.y_mean = np.zeros(y_all.shape[1],  dtype=np.float32)
            self.y_std  = np.ones(y_all.shape[1],   dtype=np.float32)

        self.input_dim = x_all.shape[1]

        # --- normalise and store per-task tensors ---
        self.train_tasks = []
        for x, y in zip(train_xs, train_ys):
            X = torch.tensor((x - self.x_mean) / self.x_std)
            Y = torch.tensor((y - self.y_mean) / self.y_std)
            self.train_tasks.append((X, Y))

        # --- test task (mustard) ---
        _, xp_test, yp_test = self.TEST_OBJECT
        x_test_raw = np.load(xp_test).astype(np.float32)
        y_test_raw = np.load(yp_test).astype(np.float32)
        x_test_raw = self._append_curvature(x_test_raw)
        x_test = select_features(x_test_raw, with_llc)
        self.x_test = torch.tensor((x_test - self.x_mean) / self.x_std)
        self.y_test = torch.tensor((y_test_raw - self.y_mean) / self.y_std)

        print(f"Input dim: {self.input_dim}  |  "
              f"Train tasks: {len(self.train_tasks)}  |  "
              f"Test samples: {len(self.x_test)}")

    @property
    def norm_stats(self):
        return dict(x_mean=self.x_mean, x_std=self.x_std,
                    y_mean=self.y_mean, y_std=self.y_std)

    def sample_episode(self, task_idx, k_shot=K_SHOT, query_size=QUERY_SIZE, device='cpu'):
        """Return (support_x, support_y, query_x, query_y) for one task."""
        X, Y = self.train_tasks[task_idx]
        n    = len(X)
        idx  = torch.randperm(n)
        s_idx, q_idx = idx[:k_shot], idx[k_shot: k_shot + query_size]
        return (X[s_idx].to(device), Y[s_idx].to(device),
                X[q_idx].to(device), Y[q_idx].to(device))


# ---------------------------------------------------------------------------
# 2.  Model (same MLP backbone)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim=5, output_dim=3, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers   = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.BatchNorm1d(h)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


    def forward_with_params(self, x, params):
        """
        Functional forward pass using an external parameter list.
        Allows us to compute gradients w.r.t. `params` without modifying
        the model's actual .parameters().
        """
        param_iter = iter(params)

        def _apply(module, x):
            if isinstance(module, nn.Linear):
                w = next(param_iter)
                b = next(param_iter)
                return torch.nn.functional.linear(x, w, b)
            elif isinstance(module, nn.BatchNorm1d):
                w = next(param_iter)
                b = next(param_iter)
                # use running stats from the real module (eval-style)
                return torch.nn.functional.batch_norm(
                    x, module.running_mean, module.running_var,
                    w, b, training=True, eps=module.eps, momentum=module.momentum)
            elif isinstance(module, nn.ReLU):
                return torch.relu(x)
            return x

        out = x
        for m in self.net:
            out = _apply(m, out)
        return out


# ---------------------------------------------------------------------------
# 3.  MAML Trainer
# ---------------------------------------------------------------------------
class MAMLTrainer:
    def __init__(self, model, task_collection, device,
                 lr_inner=LR_INNER, lr_outer=LR_OUTER,
                 epochs=EPOCHS, inner_steps=INNER_STEPS,
                 k_shot=K_SHOT, query_size=QUERY_SIZE):
        self.model          = model.to(device)
        self.tasks          = task_collection
        self.device         = device
        self.lr_inner       = lr_inner
        self.epochs         = epochs
        self.inner_steps    = inner_steps
        self.k_shot         = k_shot
        self.query_size     = query_size
        self.n_tasks        = len(task_collection.train_tasks)

        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=lr_outer)
        self.loss_fn        = nn.MSELoss()

        self.meta_train_losses = []
        self.meta_val_losses   = []

    # ── inner loop: adapt a copy of params to a single task ──
    def _inner_loop(self, support_x, support_y, params):
        adapted = [p.clone() for p in params]  # task-specific copy

        for _ in range(self.inner_steps):
            pred  = self.model.forward_with_params(support_x, adapted)
            loss  = self.loss_fn(pred, support_y)
            grads = torch.autograd.grad(loss, adapted, create_graph=True,
                                        allow_unused=True)
            adapted = [p - self.lr_inner * (g if g is not None else torch.zeros_like(p))
                       for p, g in zip(adapted, grads)]
        return adapted

    # ── outer loop: one meta-update over all tasks ──
    def _meta_step(self):
        meta_loss = torch.tensor(0.0, device=self.device)
        meta_params = list(self.model.parameters())

        for task_idx in range(self.n_tasks):
            sx, sy, qx, qy = self.tasks.sample_episode(
                task_idx, self.k_shot, self.query_size, self.device)

            adapted_params = self._inner_loop(sx, sy, meta_params)

            q_pred    = self.model.forward_with_params(qx, adapted_params)
            task_loss = self.loss_fn(q_pred, qy)
            meta_loss = meta_loss + task_loss

        meta_loss = meta_loss / self.n_tasks
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        return meta_loss.item()

    # ── quick eval on the held-out test task (no fine-tuning) ──
    @torch.no_grad()
    def _val_loss(self):
        self.model.eval()
        pred = self.model(self.tasks.x_test.to(self.device))
        loss = self.loss_fn(pred, self.tasks.y_test.to(self.device))
        self.model.train()
        return loss.item()

    def train(self):
        best_val   = float('inf')
        print(f"\n{'='*60}")
        print("META-TRAINING  (MAML)")
        print(f"  Tasks: {self.n_tasks}  |  K-shot: {self.k_shot}  |  "
              f"Inner steps: {self.inner_steps}")
        print(f"  lr_inner={self.lr_inner}  lr_outer={self.meta_optimizer.param_groups[0]['lr']}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = self._meta_step()
            val_loss   = self._val_loss()

            self.meta_train_losses.append(train_loss)
            self.meta_val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), "best_maml_model.pth")

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch [{epoch:4d}/{self.epochs}]  "
                      f"meta-train loss: {train_loss:.6f}  "
                      f"val loss: {val_loss:.6f}")

        print(f"\nMeta-training complete.  Best val loss: {best_val:.6f}")
        self._save_curves()
        return self.model

    def _save_curves(self):
        ep = np.arange(1, len(self.meta_train_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(ep, self.meta_train_losses, label='Meta-Train Loss', linewidth=2)
        plt.plot(ep, self.meta_val_losses,   label='Val Loss (no FT)', linewidth=2)
        plt.xlabel('Meta-Epoch');  plt.ylabel('MSE Loss')
        plt.title('MAML – Meta-Training & Validation Loss')
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig('data/maml_loss_curves.png', dpi=300)
        print("Saved loss curves → data/maml_loss_curves.png")
        plt.close()


# ---------------------------------------------------------------------------
# 4.  Test-time fine-tuning + evaluation
# ---------------------------------------------------------------------------
class MAMLEvaluator:
    """
    1. Fine-tune the meta-learned init on a small support set from the test task.
    2. Evaluate on the remaining query samples.
    """

    def __init__(self, model, device, task_collection,
                 finetune_steps=FINETUNE_STEPS, finetune_lr=FINETUNE_LR,
                 k_shot=K_SHOT):
        self.model          = model.to(device)
        self.device         = device
        self.tasks          = task_collection
        self.finetune_steps = finetune_steps
        self.finetune_lr    = finetune_lr
        self.k_shot         = k_shot
        self.loss_fn        = nn.MSELoss()

    def fine_tune_and_evaluate(self):
        X_test = self.tasks.x_test.to(self.device)
        Y_test = self.tasks.y_test.to(self.device)
        n      = len(X_test)

        # split test set into support / query
        idx     = torch.randperm(n)
        s_idx   = idx[:self.k_shot]
        q_idx   = idx[self.k_shot:]
        sx, sy  = X_test[s_idx], Y_test[s_idx]
        qx, qy  = X_test[q_idx], Y_test[q_idx]

        # --- zero-shot (no fine-tuning) ---
        self.model.eval()
        with torch.no_grad():
            loss_0shot = self.loss_fn(self.model(qx), qy).item()

        # --- fine-tune on support set ---
        ft_model = copy.deepcopy(self.model).train()
        opt      = torch.optim.SGD(ft_model.parameters(), lr=self.finetune_lr)
        for _ in range(self.finetune_steps):
            opt.zero_grad()
            loss = self.loss_fn(ft_model(sx), sy)
            loss.backward()
            opt.step()

        # --- evaluate fine-tuned model on query set ---
        ft_model.eval()
        with torch.no_grad():
            y_pred_norm = ft_model(qx)

        loss_ft = self.loss_fn(y_pred_norm, qy).item()

        # denormalise
        ys = torch.tensor(self.tasks.y_std,  device=self.device)
        ym = torch.tensor(self.tasks.y_mean, device=self.device)
        y_pred = (y_pred_norm * ys + ym).cpu().numpy()
        y_true = (qy          * ys + ym).cpu().numpy()

        mse            = np.mean((y_pred - y_true) ** 2)
        mae            = np.mean(np.abs(y_pred - y_true))
        mse_per_output = np.mean((y_pred - y_true) ** 2, axis=0)

        print("\n" + "=" * 60)
        print("TEST SET EVALUATION  (MAML + fine-tuning)")
        print("=" * 60)
        print(f"  Support samples (fine-tune): {len(sx)}")
        print(f"  Query  samples  (eval):      {len(qx)}")
        print(f"  Normalised MSE  — zero-shot:  {loss_0shot:.6f}")
        print(f"  Normalised MSE  — after FT:   {loss_ft:.6f}")
        print(f"  MSE  (denorm):  {mse:.6f}")
        print(f"  MAE  (denorm):  {mae:.6f}")
        print(f"  dx_MSE: {mse_per_output[0]:.6f}")
        print(f"  dy_MSE: {mse_per_output[1]:.6f}")
        print(f"  dθ_MSE: {mse_per_output[2]:.6f}")
        print("=" * 60 + "\n")

        return dict(mse=mse, mae=mae, mse_per_output=mse_per_output,
                    predictions=y_pred, ground_truth=y_true,
                    loss_0shot=loss_0shot, loss_ft=loss_ft,
                    finetuned_model=ft_model)


# ---------------------------------------------------------------------------
# 5.  Prediction on a novel object
# ---------------------------------------------------------------------------
class Predicting:
    def __init__(self, model, device, norm_stats, with_llc=False):
        self.model    = model.to(device).eval()
        self.device   = device
        self.x_mean   = norm_stats['x_mean']
        self.x_std    = norm_stats['x_std']
        self.y_mean   = norm_stats['y_mean']
        self.y_std    = norm_stats['y_std']
        self.with_llc = with_llc

    @staticmethod
    def _mesh_to_pcd(obj_name, num_points=1000):
        pcd = ObjectPointCloud(
            f"assets/{obj_name}/textured.obj",
            z_height=0.0, num_points=num_points,
            thickness=0.001, sort_by_angle=True,
        )
        return pcd.points, pcd.normals, pcd.angles

    def predict(self, obj_name, num_points=1000, output_file="data/prediction.npy"):
        points, normals, angles = self._mesh_to_pcd(obj_name, num_points)

        if self.with_llc:
            curvatures = get_local_curvature(points, WINDOWSIZE)
            features   = np.column_stack([points[:, 0], points[:, 1],
                                          normals[:, 0], normals[:, 1],
                                          angles, curvatures]).astype(np.float32)
        else:
            features = np.column_stack([points[:, 0], points[:, 1],
                                        normals[:, 0], normals[:, 1],
                                        angles]).astype(np.float32)

        features_norm = (features - self.x_mean) / self.x_std
        x_tensor      = torch.tensor(features_norm).to(self.device)

        with torch.no_grad():
            y_pred_norm = self.model(x_tensor)

        y_pred = y_pred_norm.cpu().numpy() * self.y_std + self.y_mean
        output = np.hstack([points[:, :2], y_pred[:, :2]])   # (N, 4): px,py,dx,dy
        np.save(output_file, output)
        print(f"Saved predictions → {output_file}  shape={output.shape}")
        return output


# ---------------------------------------------------------------------------
# 6.  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── 1. Load tasks ──
    tasks = TaskCollection(normalization=NORMALIZATION, with_llc=WITH_LLC)

    # ── 2. Build meta-model ──
    model = MLP(input_dim=tasks.input_dim, output_dim=OUTPUT_DIM)

    # ── 3. Meta-train with MAML ──
    trainer = MAMLTrainer(
        model, tasks, device,
        lr_inner=LR_INNER, lr_outer=LR_OUTER,
        epochs=EPOCHS, inner_steps=INNER_STEPS,
        k_shot=K_SHOT, query_size=QUERY_SIZE,
    )
    model = trainer.train()

    # ── 4. Load best checkpoint ──
    model.load_state_dict(torch.load("best_maml_model.pth",
                                     map_location=device, weights_only=True))

    # ── 5. Fine-tune on few support samples & evaluate ──
    evaluator = MAMLEvaluator(model, device, tasks,
                              finetune_steps=FINETUNE_STEPS,
                              finetune_lr=FINETUNE_LR,
                              k_shot=K_SHOT)
    results = evaluator.fine_tune_and_evaluate()

    # ── 6. Generate predictions using the fine-tuned model ──
    predictor = Predicting(results['finetuned_model'], device,
                           tasks.norm_stats, with_llc=WITH_LLC)
    predictor.predict(NOVEL_OBJ, num_points=1000, output_file="data/prediction.npy")