import torch
import torch.nn as nn

# --- THE RAY TRACER ---
class MirrorRayTracer(nn.Module):
    def __init__(self, target_x=10.0):
        super().__init__()
        self.target_x = target_x
        # Light comes from +Y (0, 1, 0)
        self.register_buffer('v_in', torch.tensor([0.0, 1.0, 0.0]))

    def forward(self, coords, deformation):
        # Base Shape: 45-degree tilted plane (y = -x)
        # This is the "bias" geometry that sends light from +Y to +X roughly.
        x = coords[:, 0:1]
        base_shape = -x

        y_surf = base_shape + deformation

        ones = torch.ones(y_surf.shape).to(y_surf.device)

        # Surface Normals (Gradients)
        grads = torch.autograd.grad(
            outputs=y_surf, inputs=coords,
            grad_outputs=ones.clone(),
            create_graph=True,
            retain_graph=True
        )[0]
        dy_dx, dy_dz = grads[:, 0:1], grads[:, 1:2]
        
        # Intersection Point
        x_surf, z_surf = coords[:, 0:1], coords[:, 1:2]
        
        # Reflection Vector (Snell's Law / Law of Reflection)
        # Normal = (-dy/dx, 1, -dy/dz) normalized
        n_unnorm = torch.cat([-dy_dx, ones.clone(), -dy_dz], dim=1)
        n_vec = torch.nn.functional.normalize(n_unnorm, dim=1)
        
        v_in = self.v_in.unsqueeze(0).expand(coords.shape[0], 3)
        dot = torch.sum(v_in * n_vec, dim=1, keepdim=True)
        v_refl = v_in - 2 * dot * n_vec
        
        # Intersect with Target Plane (x = target_x)
        t = (self.target_x - x_surf) / (v_refl[:, 0:1] + 1e-6)
        
        y_target = y_surf + t * v_refl[:, 1:2]
        z_target = z_surf + t * v_refl[:, 2:3]
        
        return torch.cat([y_target, z_target], dim=1), grads
