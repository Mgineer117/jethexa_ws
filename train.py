import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


# ==========================================
# 1. DATASET & FINITE DIFFERENCE GENERATION
# ==========================================
class HexapodDynamicsDataset(Dataset):
    def __init__(self, directory=".", dt=0.1, max_files=None):
        self.dt = dt
        self.x_data = []
        self.x_dot_data = []
        self.u_data = []

        files = glob.glob(os.path.join(directory, "*.npz"))
        valid_count = 0

        for f in files:
            if max_files is not None and valid_count >= max_files:
                break

            try:
                states = np.load(f)["states"]  # [-315:-14]
                controls = np.load(f)["controls"]  # [-315:-14]

                # Sanity checks
                if np.isnan(states).any() or states.shape[0] < 300:
                    continue

                # Drop Base Pos Z (index 2) to convert 24D -> 23D state
                states_wrapped = np.delete(states, 2, axis=1)

                states_unwrapped = states_wrapped.copy()
                states_unwrapped[:, 2:] = np.unwrap(states_unwrapped[:, 2:], axis=0)

                # =========================================================
                # SECOND ORDER FINITE DIFFERENCE
                # =========================================================
                x_dot_full = np.zeros_like(states_unwrapped)

                # 1. First point: Second-Order Forward Difference
                # Formula: (-3*x_0 + 4*x_1 - x_2) / 2dt
                x_dot_full[0, :] = (
                    -3 * states_unwrapped[0, :]
                    + 4 * states_unwrapped[1, :]
                    - states_unwrapped[2, :]
                ) / (2 * self.dt)

                # 2. Interior points: Second-Order Central Difference
                # Formula: (x_{t+1} - x_{t-1}) / 2dt
                x_dot_full[1:-1, :] = (
                    states_unwrapped[2:, :] - states_unwrapped[:-2, :]
                ) / (2 * self.dt)

                # 3. Last point: Second-Order Backward Difference
                # Formula: (3*x_N - 4*x_{N-1} + x_{N-2}) / 2dt
                x_dot_full[-1, :] = (
                    3 * states_unwrapped[-1, :]
                    - 4 * states_unwrapped[-2, :]
                    + states_unwrapped[-3, :]
                ) / (2 * self.dt)

                # Maintain your original slicing (dropping the final timestep)
                x = states_wrapped[:-1, :]
                x_dot = x_dot_full[:-1, :]

                self.x_data.append(x)
                self.x_dot_data.append(x_dot)
                self.u_data.append(controls[:-1])

                valid_count += 1

            except Exception as e:
                print(f"Failed to process {os.path.basename(f)}: {e}")

        if valid_count == 0:
            print(f"Warning: No valid files found in {directory}")
            self.x_data = torch.empty((0, 23))
            self.x_dot_data = torch.empty((0, 23))
            self.u_data = torch.empty((0, 18))
        else:
            self.x_data = torch.tensor(np.vstack(self.x_data), dtype=torch.float32)
            self.x_dot_data = torch.tensor(
                np.vstack(self.x_dot_data), dtype=torch.float32
            )
            self.u_data = torch.tensor(np.vstack(self.u_data), dtype=torch.float32)

        print(
            f"[{directory}] Loaded {valid_count} trajectories -> {self.x_data.shape[0]} total transitions."
        )

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.x_dot_data[idx], self.u_data[idx]


# ==========================================
# 2. NEURAL NETWORK B(x) FORMULATION
# ==========================================
class DataDrivenDynamics(nn.Module):
    def __init__(self, state_dim=23, base_dim=5, u_dim=18, angle_idx=[2, 3, 4]):
        """
        angle_idx: The indices of the state vector that represent angles.
        For a 23D hexapod state (Z removed), indices 2, 3, 4 are roll, pitch, yaw.
        """
        super(DataDrivenDynamics, self).__init__()
        self.base_dim = base_dim
        self.u_dim = u_dim
        self.angle_idx = sorted(angle_idx)

        # 1. Expand the input dimension:
        # For each angle, we remove 1 raw feature and add 2 (cos and sin).
        # Therefore, new dimension = original + number of angles.
        self.nn_input_dim = state_dim + len(self.angle_idx)

        # 2. Update the first linear layer to accept the encoded state
        self.jacobian_net = nn.Sequential(
            nn.Linear(self.nn_input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, base_dim * u_dim),
        )

    def encode_angles(self, x: torch.Tensor):
        """Replaces raw angles with their cos and sin embeddings."""
        if not self.angle_idx:
            return x

        components = []
        current_idx = 0

        for idx in self.angle_idx:
            # Append preceding non-angle features
            if current_idx < idx:
                components.append(x[:, current_idx:idx])

            # Extract the angle and compute sin/cos
            theta = x[:, idx : idx + 1]
            components.append(torch.cos(theta))
            components.append(torch.sin(theta))

            current_idx = idx + 1

        # Append any remaining features after the last angle (e.g., joint states)
        if current_idx < x.shape[1]:
            components.append(x[:, current_idx:])

        # Concatenate along the feature dimension (dim=1)
        return torch.cat(components, dim=1)

    def forward(self, x, u):
        batch_size = x.shape[0]

        # 1. Encode the state before passing it to the weights
        x_encoded = self.encode_angles(x)

        # 2. Pass the ENCODED state to the network
        J_flat = self.jacobian_net(x_encoded)

        # 3. The rest of the B(x) formulation remains identical
        J_base = J_flat.view(batch_size, self.base_dim, self.u_dim)
        v_base_pred = torch.bmm(J_base, u.unsqueeze(-1)).squeeze(-1)
        v_joints_pred = u

        # Output remains the original 23D state derivative
        x_dot_pred = torch.cat([v_base_pred, v_joints_pred], dim=-1)

        return x_dot_pred


# ==========================================
# 3. EVALUATION AND PLOTTING
# ==========================================
def evaluate_and_plot(model, test_dir, device):
    print("\n--- Running Test Evaluation (Best Model) ---")
    test_dataset = HexapodDynamicsDataset(directory=test_dir, max_files=1)

    if len(test_dataset) == 0:
        print("No valid test data found to plot.")
        return

    model.eval()
    with torch.no_grad():
        x, x_dot_true, u = (
            test_dataset.x_data,
            test_dataset.x_dot_data,
            test_dataset.u_data,
        )
        x, u = x.to(device), u.to(device)
        x_dot_pred = model(x, u).cpu().numpy()
        x_dot_true = x_dot_true.numpy()

    labels = [
        r"$\dot{p}_x$",
        r"$\dot{p}_y$",
        r"$\dot{\phi}$",
        r"$\dot{\theta}$",
        r"$\dot{\psi}$",
    ]

    # Increase figure height slightly for larger fonts
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    mse_val = np.mean((x_dot_pred - x_dot_true) ** 2)
    fig.suptitle(
        f"Prediction on Test Trajectory (MSE: {mse_val:.6f})",
        fontsize=20,  # Larger Title
    )

    time_axis = np.arange(x_dot_true.shape[0]) * 0.1

    for i in range(5):
        axs[i].scatter(
            time_axis, x_dot_true[:, i], label="Ground Truth", color="blue", alpha=0.6
        )
        axs[i].scatter(
            time_axis,
            x_dot_pred[:, i],
            label="Prediction",
            color="red",
            alpha=0.6,
            marker="x",
            s=40,  # Larger marker size
        )
        axs[i].plot(
            time_axis,
            x_dot_pred[:, i],
            color="orangered",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
        )

        # Font size updates
        axs[i].set_ylabel(labels[i], fontsize=18)
        axs[i].tick_params(axis="both", which="major", labelsize=12)
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(loc="upper right", fontsize=14)

    axs[-1].set_xlabel("Time (s)", fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
    plt.savefig("test_prediction_plot.svg", dpi=300)


def plot_learning_curves(train_losses, val_losses, lrs):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, "b-", label="Training Loss")
    ax1.plot(epochs, val_losses, "r-", label="Validation Loss")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Add learning rate to the secondary axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, lrs, "g--", alpha=0.5, label="Learning Rate")
    ax2.set_ylabel("Learning Rate", color="g", fontsize=12)

    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.85), fontsize=12)
    plt.title("Training Metrics Over Epochs", fontsize=14)
    plt.tight_layout()
    plt.savefig("training_curves.svg", dpi=300)


# ==========================================
# 4. MAIN TRAINING LOOP
# ==========================================
def train_dynamics_model(
    data_dir="hexapod_data", epochs=500, batch_size=512, initial_lr=1e-3, min_lr=1e-5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_dir = os.path.join(data_dir, "training")
    val_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "test")

    print("\n--- Loading Training Data ---")
    train_dataset = HexapodDynamicsDataset(directory=train_dir, max_files=20)
    print("\n--- Loading Validation Data ---")
    val_dataset = HexapodDynamicsDataset(directory=val_dir, max_files=5)

    if len(train_dataset) == 0:
        print("Training dataset is empty. Exiting.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = DataDrivenDynamics().to(device)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    # Cosine Annealing Learning Rate Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    # Huber Loss to prevent finite-difference noise spikes from dominating gradients
    criterion = nn.MSELoss()

    history_train_loss = []
    history_val_loss = []
    history_lr = []
    best_val_loss = float("inf")
    best_model_path = "J_varpi_weights_best.pth"

    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        # 1. Training Phase
        model.train()
        train_loss = 0.0

        # Define your custom weights at the top of your train_dynamics_model function
        weight_x = 1.0  # Put 10x more emphasis on p_x and p_y
        weight_y = 1.0  # Put 10x more emphasis on p_x and p_y
        weight_rot = 1.0  # Standard weight for roll, pitch, yaw

        for x, x_dot, u in train_loader:
            x, x_dot, u = x.to(device), x_dot.to(device), u.to(device)

            optimizer.zero_grad()
            x_dot_pred = model(x, u)
            # 1. Slice the tensors to calculate separate losses
            # dims 0:2 are (v_x, v_y), dims 2:5 are (omega_roll, omega_pitch, omega_yaw)
            loss_x = criterion(x_dot_pred[:, 0:1], x_dot[:, 0:1])
            loss_y = criterion(x_dot_pred[:, 1:2], x_dot[:, 1:2])
            loss_rot = criterion(x_dot_pred[:, 2:5], x_dot[:, 2:5])

            # 2. Apply weights and sum them together
            loss = (weight_x * loss_x) + (weight_y * loss_y) + (weight_rot * loss_rot)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_dataset)
        history_train_loss.append(train_loss)

        # 2. Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, x_dot, u in val_loader:
                x, x_dot, u = x.to(device), x_dot.to(device), u.to(device)
                x_dot_pred = model(x, u)

                loss_x = criterion(x_dot_pred[:, 0:1], x_dot[:, 0:1])
                loss_y = criterion(x_dot_pred[:, 1:2], x_dot[:, 1:2])
                loss_rot = criterion(x_dot_pred[:, 2:5], x_dot[:, 2:5])

                loss = (
                    (weight_x * loss_x) + (weight_y * loss_y) + (weight_rot * loss_rot)
                )

                val_loss += loss.item() * x.size(0)

        if len(val_dataset) > 0:
            val_loss /= len(val_dataset)
            history_val_loss.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.jacobian_net.state_dict(), best_model_path)
                saved_status = "(Saved Best)"
            else:
                saved_status = ""
        else:
            history_val_loss.append(0.0)
            saved_status = ""

        # Record current LR and Step the Scheduler
        current_lr = scheduler.get_last_lr()[0]
        history_lr.append(current_lr)
        scheduler.step()

        # Print less frequently since we have 1000 epochs
        if (epoch + 1) % 50 == 0 or epoch == 0 or saved_status:
            print(
                f"Epoch [{epoch+1:04d}/{epochs}] | LR: {current_lr:.2e} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} {saved_status}"
            )

    print("\nTraining complete!")
    print(f"Best Validation Loss: {best_val_loss:.6f}")

    plot_learning_curves(history_train_loss, history_val_loss, history_lr)

    model.jacobian_net.load_state_dict(torch.load(best_model_path))
    evaluate_and_plot(model, test_dir, device)


if __name__ == "__main__":
    train_dynamics_model("hexapod_data", epochs=3000)
