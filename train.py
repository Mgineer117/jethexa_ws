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
            theta = x[:, idx:idx+1]
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
