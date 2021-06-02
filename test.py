import torch
from libs.curve_rays import CurveModel
from libs.other_helpers import unit_vector, random_on_unit_sphere
import matplotlib.pyplot as plt

ckpt_path = 'logs/spheres_condensation_curved/340000.tar'
near = 2.
far = 6.
N_samples = 64
N_rays = 100
model = CurveModel()
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt['curver'])
model.eval()

rays_o = random_on_unit_sphere((N_rays, 3))*4
# rays_d = -1 * torch.rand((N_rays, 3)) + 1 # This is distributed -1 to 1
rays_d = unit_vector(-rays_o)

t_vals = torch.linspace(0., 1., steps=N_samples)
z_vals = near * (1. - t_vals) + far * t_vals
z_vals = z_vals.expand([N_rays, N_samples])
target = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

input_batch = torch.cat((rays_o, rays_d), dim=-1).unsqueeze(1)
input_batch = input_batch.expand(N_rays, N_samples, 6)
input_batch = torch.cat((input_batch, z_vals.unsqueeze(-1)), dim=-1)

o = model(input_batch)

o = o.detach().cpu().numpy()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for i in o:
    ax.plot(i[:, 0], i[:, 1], i[:, 2])
plt.show()
