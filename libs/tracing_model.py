import torch
import torch.nn as nn
from libs.run_nerf_helpers import gather_batch


class IoRModel(nn.Module):
    def __init__(self):
        super(IoRModel, self).__init__()
        self.fc1 = nn.Linear(3, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc_n = nn.Linear(512, 1)
        self.fc_grad = nn.Linear(512, 3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        n = self.sigmoid(self.fc_n(x)) + 1
        gradient = self.fc_grad(x)
        return n, gradient


class IoRModelTrivial(nn.Module):
    def __init__(self):
        super(IoRModelTrivial, self).__init__()

    def forward(self, x):
        return 1, torch.zeros_like(x)


class EvolutionModel(nn.Module):
    def __init__(self, ior_model, step_size, bounds_box):
        super(EvolutionModel, self).__init__()

        self.ior_model = ior_model
        self.step_size = step_size
        self.n_steps = 2 / self.step_size
        self.bound_box = bounds_box
        # self.n_samples = n_samples

    def forward(self, x0, v0, z_vals):

        evolution = self.evolve(x0, v0)

        x_hist = evolution['x_hist']
        # m_hist = evolution['m_hist']
        d = evolution['distances']

        # self.z_vals = self.z_vals.expand([N_rays, self.n_samples]).unsqueeze(-1)
        final_coords = self.sample_rays(x_hist, d, z_vals)
        assert not final_coords.isnan().any()
        return final_coords

    @staticmethod
    def sample_rays(r_hist, distances, z_vals):
        tminusd = z_vals - distances.unsqueeze(1)
        tminusd_pos = tminusd.clone()
        tminusd_neg = tminusd.clone()
        tminusd_pos[tminusd_pos <= 0] = 10
        tminusd_neg[tminusd_neg > 0] = -10

        idx_top_pos = torch.topk(tminusd_pos, k=1, largest=False, sorted=False)  # Take the smallest
        idx_top_neg = torch.topk(tminusd_neg, k=1, largest=True, sorted=False)  # Take the largest

        indices = torch.cat([idx_top_pos.indices, idx_top_neg.indices], dim=-1)
        values = torch.cat([idx_top_pos.values, idx_top_neg.values], dim=-1)

        # idx_top = torch.topk(tminusd, k=2, largest=False, sorted=False)
        coords = gather_batch(r_hist, indices)
        m = (coords[:, :, 1] - coords[:, :, 0]) / z_vals

        m = m / m.norm(dim=-1, keepdim=True)
        final_coords = coords[:, :, 0] + values[..., :1].repeat_interleave(3, dim=-1) * m

        return final_coords

    # plot_tetra(self.graph, tetra_idx[:, None], rp[:, None, :], m[:, None, :])
    def evolve(self, x0, v0):
        x = x0.clone()
        v = v0.clone()

        x_hist = [x]
        v_hist = [v]
        distances = [torch.zeros(x.shape[0], device=x.device)]

        active = torch.ones(x0.shape[0], device=x0.device, dtype=torch.bool)
        x_out = torch.zeros_like(x0, requires_grad=True)
        v_out = torch.zeros_like(v0, requires_grad=True)

        i = 0
        while i < self.n_steps:
            ior, ior_grad = self.ior_model(x)
            v_half = v + 0.5 * self.step_size * ior * ior_grad
            x_next = x + v_half * self.step_size
            escaped = active & ~self.is_inside(x_next)
            active = ~escaped

            escaped = escaped.unsqueeze(1).repeat(1, 3)

            x_out = torch.where(escaped, x, x_out)
            v_out = torch.where(escaped, v, v_out)

            if not active.any():
                break

            d = torch.norm(x_next - x, dim=1)

            x = x_next
            ior_next, ior_grad_next = self.ior_model(x_next)
            v = v_half + 0.5 * self.step_size * ior_next * ior_grad_next
            x_hist.append(x)
            v_hist.append(v)
            distances.append(d)

            i += 1

        x_hist = torch.stack(x_hist, dim=1)
        v_hist = torch.stack(v_hist, dim=1)
        distances = torch.stack(distances, dim=1)

        distances = torch.cumsum(distances, dim=1)

        return {'x_hist': x_hist, 'v_hist': v_hist, 'distances': distances}

    def is_inside(self, x):
        return (self.bound_box[0][0] <= x[:, 0]) & (x[:, 0] <= self.bound_box[0][1]) & \
               (self.bound_box[1][0] <= x[:, 1]) & (x[:, 1] <= self.bound_box[1][1]) & \
               (self.bound_box[2][0] <= x[:, 2]) & (x[:, 2] <= self.bound_box[2][1])


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    near = 0
    far = 1.5
    N_samples = 2
    r0 = torch.tensor([[0.44940298795700073, 0.40466320514678955, -0.5139325261116028], [0.4494, 0.40466, -0.51393]],
                      device=device)
    m0 = torch.tensor([[-0.5564, -0.5963, 0.5787], [-0.5564, -0.5963, 0.5787]], dtype=torch.float32, device=device)
    m0 = m0 / m0.norm(dim=1, keepdim=True)

    bounds_box = [(-1, 1), (-1, 1), (-1, 1)]

    ior_model = IoRModelTrivial()
    ior_model.to(device)
    step_size = 0.1

    ev = EvolutionModel(ior_model, step_size, bounds_box)
    vals = ev.evolve(r0, m0)

    # t_vals = torch.linspace(0.01, 1., steps=64)
    # z_vals = near * (1. - t_vals) + far * t_vals
    # z_vals = z_vals.to(device)
    # z_vals = z_vals.expand([1, 64]).unsqueeze(-1)
    # aa = ev(r0, m0, z_vals)

    # crit = nn.MSELoss()
    # real = torch.randn_like(f)
    # loss = crit(f, real)
    # loss.backward()
