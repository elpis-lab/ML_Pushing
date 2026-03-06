import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

NORMALIZE = True
SECOND_ORDER = True

K_SHOT = 64
QUERY_SIZE = 128

INPUT_DIM = 3       # px, py, angle
OUTPUT_DIM = 3      # dx, dy, dtheta
HIDDEN_DIMS = [256, 128, 64]

LR_INNER = 0.02     # alpha, adaptation step size
LR_OUTER = 5e-4     # beta, meta step size
INNER_STEP = 3      # 1-5, should be small
BATCH_TASKS = 8     # number of tasks per meta step
META_EPOCHS = 1000

WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.0

FINETUNE_STEPS = 200
FINETUNE_LR = 0.03
FINETUNE_KSHOT = 50

NOVEL_OBJ = "mustard_bottle_flipped"


class Tasks:
    Dtr = [
        ("apple",           "data/x_apple_1000.npy",                   "data/y_apple_1000.npy"),
        ("banana",          "data/x_banana_flipped_1000.npy",          "data/y_banana_flipped_1000.npy"),
        ("bowl",            "data/x_bowl_1000.npy",                    "data/y_bowl_1000.npy"),
        ("box_1",           "data/x_box_1_1000.npy",                   "data/y_box_1_1000.npy"),
        ("chips_can",       "data/x_chips_can_1000.npy",               "data/y_chips_can_1000.npy"),
        ("circle_1",        "data/x_circle_1_1000.npy",                "data/y_circle_1_1000.npy"),
        ("mug",             "data/x_mug_1000.npy",                     "data/y_mug_1000.npy"),
        ("orange",          "data/x_orange_1000.npy",                  "data/y_orange_1000.npy"),
        ("pudding_box",     "data/x_pudding_box_1000.npy",             "data/y_pudding_box_1000.npy"),
        ("sugar_box",       "data/x_sugar_box_1000.npy",               "data/y_sugar_box_1000.npy"),
        ("tomato_soup_can", "data/x_tomato_soup_can_1000.npy",         "data/y_tomato_soup_can_1000.npy"),
        ("tuna_fish_can",   "data/x_tuna_fish_can_1000.npy",           "data/y_tuna_fish_can_1000.npy"),
        # ("mustard_bottle",  "data/x_mustard_bottle_flipped_1000.npy",  "data/y_mustard_bottle_flipped_1000.npy"),
        ("cracker_box",     "data/x_cracker_box_flipped_1000.npy",     "data/y_cracker_box_flipped_1000.npy"),
    ]

    Dts = (
        NOVEL_OBJ,  f"data/x_{NOVEL_OBJ}_1000.npy",  f"data/y_{NOVEL_OBJ}_1000.npy"
    )

    def __init__(self):
        x_train, y_train = [], []
        for obj, x, y in self.Dtr:
            x_train.append(np.load(x).astype(np.float32))
            y_train.append(np.load(y).astype(np.float32))
        
        x_all = np.concatenate(x_train, axis=0)
        y_all = np.concatenate(y_train, axis=0)

        x_test = np.load(self.Dts[1]).astype(np.float32)
        y_test = np.load(self.Dts[2]).astype(np.float32)

        if NORMALIZE:
            self.x_mean = x_all.mean(0)
            self.y_mean = y_all.mean(0)
            self.x_std = x_all.std(0) + 1e-8
            self.y_std = y_all.std(0) + 1e-8
        else:
            self.x_mean = np.zeros(INPUT_DIM, dtype=np.float32)
            self.y_mean = np.zeros(OUTPUT_DIM, dtype=np.float32)
            self.x_std = np.ones(INPUT_DIM, dtype=np.float32)
            self.y_std = np.ones(OUTPUT_DIM, dtype=np.float32)

        def x_norm(x):
            return (x - self.x_mean) / self.x_std
        
        def y_norm(y):
            return (y - self.y_mean) / self.y_std
        
        self.train_tasks = []
        for x, y in zip(x_train, y_train):
            xtr = torch.tensor(x_norm(x), dtype=torch.float32)
            ytr = torch.tensor(y_norm(y), dtype=torch.float32)
            self.train_tasks.append((xtr, ytr))

        self.x_test = torch.tensor(x_norm(x_test))
        self.y_test = torch.tensor(y_norm(y_test))

    def sample_task(self, i, device):
        xtr, ytr = self.train_tasks[i]
        indices = torch.randperm(len(xtr))
        s = indices[:K_SHOT]                        # support indices
        q = indices[K_SHOT : K_SHOT + QUERY_SIZE]   # query indices
        xs = xtr[s].to(device)
        ys = ytr[s].to(device)
        xq = xtr[q].to(device)
        yq = ytr[q].to(device)

        return xs, ys, xq, yq
    
    def sample_task_indices(self, n):
        return torch.randperm(len(self.train_tasks))[:n].tolist()
    

class Model(nn.Module):
    def __init__(self,
                 input_dim=INPUT_DIM,
                 hidden_dims=HIDDEN_DIMS,
                 output_dim=OUTPUT_DIM,
                 dropout_rate=DROPOUT_RATE):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.LayerNorm(h),
                nn.Dropout(dropout_rate),
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    def functional_forward(self, x, params, training=True):
        it = iter(params)

        def fwd(module, input):
            if isinstance(module, nn.Linear):
                return torch.nn.functional.linear(input, next(it), next(it))
            if isinstance(module, nn.ReLU):
                return torch.relu(input)
            if isinstance(module, nn.LayerNorm):
                return torch.nn.functional.layer_norm(input, module.normalized_shape, next(it), next(it), module.eps)
            if isinstance(module, nn.Dropout):
                return torch.nn.functional.dropout(input, p=module.p, training=training)
            return input
        
        output = x
        for m in self.model:
            output = fwd(m, output)

        return output
    

class MAML:
    def __init__(self, model, tasks, device):
        self.model = model.to(device)
        self.tasks = tasks
        self.device = device
        self.n_tasks = len(tasks.train_tasks)
        self.loss_fn = nn.MSELoss()
        self.meta_opt = torch.optim.Adam(params=model.parameters(),
                                         lr=LR_OUTER,
                                         weight_decay=WEIGHT_DECAY)
        self.train_losses = []
        self.val_losses = []

    def inner_loop(self, xs, ys, meta_params):   # adaptation, alg 2-7
        adapted_params = [mp.clone() for mp in meta_params]
        for i in range(INNER_STEP):
            ys_pred = self.model.functional_forward(xs, adapted_params, training=True)    # fpxj
            support_loss = self.loss_fn(ys_pred, ys)
            grads = torch.autograd.grad(
                support_loss,
                adapted_params,
                create_graph=SECOND_ORDER,
            )
            adapted_params = [
                mp - LR_INNER * g for mp, g in zip(adapted_params, grads)
            ]
        with torch.no_grad():
            ys_pred_final = self.model.functional_forward(xs, adapted_params, training=False)
            final_support_loss = self.loss_fn(ys_pred_final, ys)

        return adapted_params, final_support_loss.item()
    
    def meta_step(self):    # meta-learning, alg 2-10
        meta_params = list(self.model.parameters())
        meta_loss = torch.zeros((), device=self.device)
        support_loss_total = 0.0
        task_indices = self.tasks.sample_task_indices(BATCH_TASKS)
        for i in task_indices:
            xs, ys, xq, yq = self.tasks.sample_task(i, self.device)
            adapted_params, support_loss = self.inner_loop(xs, ys, meta_params)   # support
            support_loss_total += support_loss
            yq_pred = self.model.functional_forward(xq, adapted_params, training=False)  # query
            query_loss = self.loss_fn(yq_pred, yq)
            meta_loss += query_loss
        meta_loss = meta_loss / len(task_indices)
        support_loss_total = support_loss_total / len(task_indices)
        self.meta_opt.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.meta_opt.step()

        return support_loss_total, meta_loss.item()
    
    def train(self):
        best_val = float('inf')
        for epoch in range(1, META_EPOCHS + 1):
            self.model.train()
            support_loss, query_loss = self.meta_step()

            self.train_losses.append(support_loss)
            self.val_losses.append(query_loss)

            if query_loss < best_val:
                best_val = query_loss
                torch.save(self.model.state_dict(), "best_maml.pth")
            
            if epoch % 50 == 0 or epoch == 1:
                print(f"[{epoch:4d}/{META_EPOCHS} | tr-loss: {support_loss:.6f} | val-loss: {query_loss:.6f}]")
        print(f"\nBest val loss: {best_val:.6f}")
        self.plot_loss_curves()

    # offline evaluation currently
    # can update with online adaptation with live push data collecting
    def evaluate(self,
                 finetune_steps=FINETUNE_STEPS,
                 finetune_lr=FINETUNE_LR,
                 k_shot=FINETUNE_KSHOT):
        xts = self.tasks.x_test.to(self.device)
        yts = self.tasks.y_test.to(self.device)
        indices = torch.randperm(len(xts))
        s = indices[:k_shot]
        q = indices[k_shot:]
        xs = xts[s]
        ys = yts[s]
        xq = xts[q]
        yq = yts[q]
        self.model.eval()
        with torch.no_grad():
            loss_0 = self.loss_fn(self.model(xq), yq).item()
        finetune = copy.deepcopy(self.model).train()    # original meta-model
        opt = torch.optim.SGD(finetune.parameters(), lr=finetune_lr, weight_decay=WEIGHT_DECAY)
        for i in range(finetune_steps):     # training raw meta-model
            opt.zero_grad()
            self.loss_fn(finetune(xs), ys).backward()
            opt.step()

        finetune.eval()
        with torch.no_grad():   # evaluation with trained meta-model
            y_pred_norm = finetune(xq)
        loss_ft = self.loss_fn(y_pred_norm, yq).item()

        y_std = torch.tensor(self.tasks.y_std, device=self.device)
        y_mean = torch.tensor(self.tasks.y_mean, device=self.device)
        y_pred = (y_pred_norm * y_std + y_mean).cpu().numpy()
        y_true = (yq * y_std + y_mean).cpu().numpy()
        mse_per = np.mean((y_pred - y_true)**2, axis=0)
        print("\nEvaluation")
        print(f"MSE zero-shot: {loss_0:.6f}")
        print(f"MSE after FT: {loss_ft:.6f}")
        print(f"dx_MSE: {mse_per[0]:.6f}")
        print(f"dy_MSE: {mse_per[1]:.6f}")
        print(f"dθ_MSE: {mse_per[2]:.6f}")

        return finetune

    def predict(self, model, x_path, output_file):
        x_raw = np.load(x_path).astype(np.float32)[:, :INPUT_DIM]
        x_norm = (x_raw - self.tasks.x_mean) / self.tasks.x_std
        x = torch.tensor(x_norm).to(self.device)
        model.eval()
        with torch.no_grad():
            y_norm = model(x).cpu().numpy()
        y_pred = y_norm * self.tasks.y_std + self.tasks.y_mean
        np.save(output_file, y_pred)
        print(f"Saved to {output_file}")

    def plot_loss_curves(self):
        epoch = np.arange(1, len(self.train_losses) + 1)
        plt.figure(figsize=(9, 5))
        plt.plot(epoch, self.train_losses, label="support_loss")
        plt.plot(epoch, self.val_losses, label="query_loss")
        plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("MAML Loss")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig("data/maml_loss.png", dpi=150)
        print("Saved to data/maml_loss.png")
        plt.close()

if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    tasks = Tasks()
    model = Model()

    trainer = MAML(model, tasks, device)
    trainer.train()

    model.load_state_dict(torch.load("best_maml.pth", map_location=device, weights_only=True))
    finetuned_model = trainer.evaluate()
    trainer.predict(finetuned_model, f"data/x_{NOVEL_OBJ}_1000.npy", "data/pred.npy")