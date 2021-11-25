import torch
import torch.nn as nn
from libs.mesh_helpers import read_mesh, wrap_to_2pi, batch_dot, gather_batch
import libs.transforms as t
from einops import rearrange


class EvolutionModel(nn.Module):
    def __init__(self, filename, n_steps, transforms):
        super(EvolutionModel, self).__init__()
        self.graph = read_mesh(filename, '3D', transforms)

        self.n_steps = n_steps
        # self.n_samples = n_samples

        self.initialized = False
        self.num_nodes = self.graph.pos.shape[0]
        self.num_tetra = self.graph.tetra.shape[1]

    def forward(self, r0, m0, z_vals):
        assert self.initialized, 'init_vars() must be called before trying to evolve.'

        evolution = self.evolve(r0, m0)

        r_hist = evolution['r_hist']
        # m_hist = evolution['m_hist']
        d = evolution['distances']

        # self.z_vals = self.z_vals.expand([N_rays, self.n_samples]).unsqueeze(-1)
        final_coords = self.sample_rays(r_hist, d, z_vals)
        return final_coords

    @staticmethod
    def sample_rays(r_hist, distances, z_vals):
        tminusd = z_vals - distances.unsqueeze(1)
        tminusd_pos = tminusd.clone()
        tminusd_neg = tminusd.clone()
        tminusd_pos[tminusd_pos < 0] = 10
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

    def to(self, *args, **kwargs):
        new_self = super(EvolutionModel, self).to(*args, **kwargs)
        # new_self.z_vals = new_self.z_vals.to(*args, **kwargs)
        new_self.graph = new_self.graph.to(*args, **kwargs)

        return new_self

    def init_vars(self, n_index):
        assert n_index.shape[0] == self.num_nodes, 'n_index must have as many values as the number of nodes'
        self.graph.n_index = nn.Parameter(n_index)
        self.update_vars()
        self.initialized = True

        return self.graph

    def update_vars(self):
        coords_vertex_tetra = rearrange(self.graph.pos[self.graph.tetra], 'v t c -> t c v')
        k = torch.cat((torch.ones((self.num_tetra, 1, 4), device=coords_vertex_tetra.device), coords_vertex_tetra), dim=1).inverse()

        ab = torch.bmm(torch.transpose(k, 1, 2), self.graph.n_index[self.graph.tetra].T.unsqueeze(-1))
        a = ab[:, 0, 0]
        b = ab[:, 1:, 0]
        n = b / b.norm(dim=1, keepdim=True)

        self.graph.n = n
        self.graph.a = a
        self.graph.b = b

    def evolve_in_tetrahedron(self, tetra_idx, rp, m):
        """ This is where the magic happens. Based on https://academic.oup.com/qjmam/article/65/1/87/1829302"""
        bs = m.shape[0]

        outside_idx = (tetra_idx == -1)
        outside_pos = rp + m  # Straight line

        a = self.graph.a[tetra_idx]
        b = self.graph.b[tetra_idx]
        # p = self.graph.pos[self.graph.tetra[:, tetra_idx]]
        n = self.graph.n[tetra_idx]
        # p = rearrange(p, 'p b c -> b p c')
        faces = self.graph.tetra_face[tetra_idx]
        faces = rearrange(faces, 'b f -> f b')

        mn = torch.cross(m, n)
        q = mn / torch.sqrt(batch_dot(mn, mn).unsqueeze(1))
        nq = torch.cross(n, q)
        mn_dot = batch_dot(m, n).unsqueeze(1)
        rc = rp - (batch_dot(rp, n) + a / b.norm(dim=1)).unsqueeze(1) * (n - (mn_dot * nq) / batch_dot(m, nq).unsqueeze(1))
        R = rc - rp

        phiR = torch.zeros((bs, 4), device=R.device)
        for it, face in enumerate(faces):
            vtx_face = self.graph.face_vertex[face]
            pos_vtx_face = self.graph.pos[vtx_face]
            i = pos_vtx_face[:, 0]
            j = pos_vtx_face[:, 1]
            k = pos_vtx_face[:, 2]

            M_L0 = (j[:, 1] - i[:, 1]) * (k[:, 2] - i[:, 2]) - (k[:, 1] - i[:, 1]) * (j[:, 2] - i[:, 2])
            M_L1 = (j[:, 2] - i[:, 2]) * (k[:, 0] - i[:, 0]) - (k[:, 2] - i[:, 2]) * (j[:, 0] - i[:, 0])
            M_L2 = (j[:, 0] - i[:, 0]) * (k[:, 1] - i[:, 1]) - (k[:, 0] - i[:, 0]) * (j[:, 1] - i[:, 1])
            M_L = torch.stack((M_L0, M_L1, M_L2), dim=1)
            Q_L = - batch_dot(i, M_L)

            c1 = - batch_dot(M_L, R)
            c2 = R.norm(dim=1) * batch_dot(M_L, m)
            c3 = batch_dot(M_L, rc) + Q_L

            # if c1 == c3:
            #    phi1 = -2*torch.atan(c1/c2)
            #    phi2 = phi1
            # else:
            phi1 = 2 * torch.atan((c2 + torch.sqrt(c1.pow(2) + c2.pow(2) - c3.pow(2))) / (c1 - c3))
            phi2 = 2 * torch.atan((c2 - torch.sqrt(c1.pow(2) + c2.pow(2) - c3.pow(2))) / (c1 - c3))
            phi1 = wrap_to_2pi(phi1)
            phi2 = wrap_to_2pi(phi2)
            phi_cat = torch.stack([phi1, phi2], dim=-1)
            phiR[:, it] = phi_cat.min(dim=1).values

        phiR = torch.nan_to_num(phiR, nan=10)
        phiE, idx_face = phiR.min(dim=1, keepdim=True)

        phiE += phiE * 1 / 100  # This is to make sure that the next point starts inside the next tetrahedron

        # New direction and position
        re = rc - torch.cos(phiE) * R + R.norm(dim=1, keepdim=True) * torch.sin(phiE) * m
        me = torch.cos(phiE) * m + torch.sin(phiE) / R.norm(dim=1, keepdim=True) * R

        # Face number
        hit_face = faces[idx_face.squeeze()].diag()  # Get the diagonal

        # Next tetrahedron
        local_idx_next = (self.graph.face_tetra[hit_face] != tetra_idx.unsqueeze(1)).nonzero(as_tuple=True)

        next_tetra = self.graph.face_tetra[hit_face][local_idx_next]

        distance = torch.norm(rp - re, dim=1)

        # re = torch.where(outside_idx.unsqueeze(-1), outside_pos, re)
        # next_tetra = torch.where(next_tetra == -1, tetra_idx, next_tetra)

        return next_tetra, re, me, distance

    def evolve(self, r0, m0):
        r = r0.clone()
        m = m0.clone()
        tetra_idx = self.find_tetrahedron_point(r0)

        i = 0
        r_hist = [r]
        m_hist = [m]
        distances = [torch.zeros(r.shape[0], device=r.device)]
        tetra_hist = [tetra_idx]
        while i < self.n_steps and (tetra_idx != -1).all():
            tetra_idx, r, m, d = self.evolve_in_tetrahedron(tetra_idx, r, m)
            r_hist.append(r)
            m_hist.append(m)
            distances.append(d)
            tetra_hist.append(tetra_idx)
            i += 1

        r_hist = torch.stack(r_hist, dim=1)
        m_hist = torch.stack(m_hist, dim=1)
        distances = torch.stack(distances, dim=1)
        tetra_hist = torch.stack(tetra_hist, dim=1)

        distances = torch.cumsum(distances, dim=1)

        return {'r_hist': r_hist, 'm_hist': m_hist, 'distances': distances, 'tetra_hist': tetra_hist}

    def find_tetrahedron_point(self, point):
        # Very much from https://stackoverflow.com/questions/25179693/how-to-check-whether-the-point-is-in-the-tetrahedron-or-not
        n_p = point.shape[0]
        orir = torch.repeat_interleave(self.graph.origin.unsqueeze(-1), n_p, dim=2)
        newp = torch.einsum('imk, kmj -> kij', self.graph.ort_tetra, point.T - orir)
        val = torch.all(newp >= 0, dim=1) & torch.all(newp <= 1, dim=1) & (torch.sum(newp, dim=1) <= 1)
        id_tet, id_p = torch.nonzero(val, as_tuple=True)

        res = - torch.ones(n_p, dtype=id_tet.dtype, device=id_tet.device)  # Sentinel value
        res[id_p] = id_tet
        return res


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mesh_type = '3D'
    filename = '../meshes/sphere_coarse.msh'
    near = 0
    far = 1.5
    N_samples = 2
    r0 = torch.tensor([[0.8, 0., 0.]], device=device)
    m0 = torch.tensor([[-1, 0.1, 0.1]], dtype=torch.float32, device=device)
    m0 = m0 / m0.norm(dim=1)
    tr = [t.TetraToEdge(remove_tetras=False), t.TetraToNeighbors(), t.TetraCoordinates()]

    ev = EvolutionModel(filename, mesh_type, n_steps=1, n_samples=N_samples, transforms=tr)
    ev.to(device)
    ev.train()

    n_index = - 0.1 * ev.graph.pos.norm(dim=1) + 1.1

    ev.compute_vars(n_index)

    f = ev(r0, m0)

    crit = nn.MSELoss()
    real = torch.randn_like(f)
    loss = crit(f, real)
    loss.backward()
