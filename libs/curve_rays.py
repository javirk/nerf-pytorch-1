import torch
import torch.nn as nn
import libs.other_helpers as u

class CurveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 128)
        self.relu = nn.ReLU
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    N_samples = 4
    near = 2.
    far = 6.
    N_rays = 1
    EPOCHS = 10

    model = CurveModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for i in range(EPOCHS):
        rays_o = u.random_on_unit_sphere((N_rays, 3)) * 4
        rays_d = -1 * torch.rand((N_rays, 3)) + 1 # This is distributed -1 to 1
        rays_d = u.unit_vector(rays_d)

        t_vals = torch.linspace(0., 1., steps=N_samples)
        z_vals = near * (1. - t_vals) + far * t_vals

        z_vals = z_vals.expand([N_rays, N_samples])

        input_batch = torch.cat((rays_o, rays_d, ))

        gt = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]