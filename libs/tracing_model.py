import torch
import torch.nn as nn
from libs.mesh_helpers import read_mesh, wrap_to_2pi, batch_dot, gather_batch
import libs.transforms as t
from einops import rearrange


class EvolutionModel(nn.Module):
    def __init__(self, folder, n_steps, transforms):
        super(EvolutionModel, self).__init__()
        self.graph = read_mesh(folder, '3D', transforms)

        self.n_steps = n_steps
        # self.n_samples = n_samples

        self.initialized = False
        self.num_nodes = self.graph.pos.shape[0]
        self.num_tetra = self.graph.tetra.shape[1]
        self.n_index = None

    def forward(self, r0, m0, z_vals):
        assert self.initialized, 'init_vars() must be called before trying to evolve.'

        evolution = self.evolve(r0, m0, z_vals.max())

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
        self.n_index = nn.Parameter(n_index)
        self.update_vars()
        self.initialized = True

        return self.graph

    def update_vars(self):
        coords_vertex_tetra = rearrange(self.graph.pos[self.graph.tetra], 'v t c -> t c v')
        k = torch.cat((torch.ones((self.num_tetra, 1, 4), device=coords_vertex_tetra.device), coords_vertex_tetra),
                      dim=1).inverse()

        ab = torch.bmm(torch.transpose(k, 1, 2), self.n_index[self.graph.tetra].T.unsqueeze(-1))
        a = ab[:, 0, 0]
        b = ab[:, 1:, 0]
        n = b / b.norm(dim=1, keepdim=True)

        self.graph.n = n
        self.graph.a = a
        self.graph.b = b

    @torch.no_grad()
    def intersect_straight_ray(self, tetra_idx, rp, m, restricted_face):
        """
        Intersect a ray with all the faces of a tetrahedron and return the parametric distance to the nearest face
        :param tetra_idx:
        :param rp:
        :param m:
        :param restricted_face: Face from which the ray comes
        :return: torch.Tensor([B, 1])
        """
        bs = m.shape[0]
        faces = self.graph.tetra_face[tetra_idx]
        faces = rearrange(faces, 'b f -> f b')

        t_all = torch.ones((bs, 4), device=faces.device, dtype=rp.dtype) * 1
        t_restricted = torch.zeros(bs, device=faces.device, dtype=rp.dtype)
        t_debug = torch.ones((bs, 4), device=faces.device, dtype=rp.dtype)
        for it, face in enumerate(faces):
            vtx_face = self.graph.face_vertex[face]
            pos_vtx_face = self.graph.pos[vtx_face]  # B x Points x Coordinates
            i = pos_vtx_face[:, 0]
            j = pos_vtx_face[:, 1]
            k = pos_vtx_face[:, 2]

            N = torch.cross(j - i, k - i, dim=1)
            D = batch_dot(N, i)
            t = (- batch_dot(N, rp) + D) / batch_dot(N, m)
            t_debug[:, it] = t
            # t_all[:, it] = torch.where(t > 0, t, t_all[:, it])
            t_restricted[:] = torch.where(face == restricted_face, t, t_restricted)
            t_all[:, it] = torch.where((t > 0) & (face != restricted_face), t, t_all[:, it])

        min_t = t_all.min(dim=1).values
        t_next = torch.max((min_t - t_restricted) / 3, t_restricted * 1.05)  # Move at least t_restricted.
        # Maybe there is no need to do the other intersections

        return t_next.unsqueeze(1)

    def evolve_in_tetrahedron(self, tetra_idx, rp, m, debug_pos=None):
        """ This is where the magic happens. Based on https://academic.oup.com/qjmam/article/65/1/87/1829302"""
        if debug_pos and debug_pos > rp.shape[0]:
            debug_pos = None

        a = self.graph.a[tetra_idx]
        b = self.graph.b[tetra_idx]
        n = self.graph.n[tetra_idx]

        mn = torch.cross(m, n)
        q = mn / torch.sqrt(batch_dot(mn, mn).unsqueeze(1))
        nq = torch.cross(n, q)
        mn_dot = batch_dot(m, n).unsqueeze(1)
        rc = rp - (batch_dot(rp, n) + a / b.norm(dim=1)).unsqueeze(1) * (
                n - (mn_dot * nq) / batch_dot(m, nq).unsqueeze(1))
        R = rc - rp
        if debug_pos:
            print(f'Tetra_idx: {tetra_idx[debug_pos]}')

        hit_face = self._find_intersection_face(rp, m, R, rc, tetra_idx, debug_pos=debug_pos)

        phiE = self._do_intersection(hit_face, R, m, rc, debug_pos=debug_pos, it='Forward')
        phiE = phiE.min(dim=1, keepdim=True).values

        # phiE += phiE / 1000

        # New direction and position
        re = rc - torch.cos(phiE) * R + R.norm(dim=1, keepdim=True) * torch.sin(phiE) * m
        assert (re.norm(dim=1) < 5).all()
        me = torch.cos(phiE) * m + torch.sin(phiE) / R.norm(dim=1, keepdim=True) * R

        # Face number
        # hit_face = faces[idx_face].diag()  # Get the diagonal

        # Next tetrahedron
        local_idx_next = (self.graph.face_tetra[hit_face] != tetra_idx.unsqueeze(1)).nonzero(as_tuple=True)
        next_tetra = self.graph.face_tetra[hit_face][local_idx_next]

        # This is to make sure that the next point starts inside the next tetrahedron.
        t = self.intersect_straight_ray(next_tetra, re, me, hit_face)
        if debug_pos:
            print(f'{t[debug_pos]=}')
        re += t * me
        assert (re.norm(dim=1) < 5).all()

        distance = torch.norm(rp - re, dim=1)

        if debug_pos:
            print('\n')

        return next_tetra, re, me, distance

    def _do_intersection(self, face, R, m, rc, debug_pos=None, it=0):
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

        phi1_equal = -2 * torch.atan(c1 / c2)
        phi2_equal = phi1_equal

        # if c1 == c3:
        #    phi1 = -2*torch.atan(c1/c2)
        #    phi2 = phi1
        # else:
        in_sqrt = torch.sqrt(c1.pow(2) + c2.pow(2) - c3.pow(2))

        phi1 = 2 * torch.atan((c2 + in_sqrt) / (c1 - c3))
        phi2 = 2 * torch.atan((c2 - in_sqrt) / (c1 - c3))
        # phi1 = 2 * torch.atan2((c2 + in_sqrt), (c1 - c3))
        # phi2 = 2 * torch.atan2((c2 - in_sqrt), (c1 - c3))

        phi1 = torch.where((phi2 == 0) & (c1 * c3 > 0), phi1_equal, phi1)
        phi2 = torch.where((phi2 == 0) & (c1 * c3 > 0), phi2_equal, phi2)

        # phi1 = 2 * torch.atan((c2 + torch.sqrt(c1.pow(2) + c2.pow(2) - c3.pow(2))) / (c1 - c3))
        # phi2 = 2 * torch.atan((c2 - torch.sqrt(c1.pow(2) + c2.pow(2) - c3.pow(2))) / (c1 - c3))
        if debug_pos:
            print(it, phi1[debug_pos].item(), phi2[debug_pos].item())
        phi1 = wrap_to_2pi(phi1)
        phi2 = wrap_to_2pi(phi2)
        phi_intersection = torch.stack([phi1, phi2], dim=-1)

        return phi_intersection

    @torch.no_grad()
    def _find_intersection_face(self, rp, m, R, rc, tetra_idx, debug_pos=None):
        faces = self.graph.tetra_face[tetra_idx]
        faces = rearrange(faces, 'b f -> f b')
        phiR = torch.zeros((rp.shape[0], 4), device=R.device, dtype=rp.dtype)

        for it, face in enumerate(faces):
            phi_intersection = self._do_intersection(face, R, m, rc, debug_pos=debug_pos, it=it)
            phiR[:, it] = phi_intersection.min(dim=1).values

        phiR = torch.nan_to_num(phiR, nan=10)
        # first_phi, _ = torch.topk(phiR, k=2, dim=1, largest=False)
        # phiE = first_phi.mean(dim=1).unsqueeze(-1)
        _, idx_face = phiR.min(dim=1, keepdim=True)

        # Face number
        hit_face = faces[idx_face.squeeze()].diag()  # Get the diagonal

        return hit_face

    # plot_tetra(self.graph, tetra_idx[:, None], rp[:, None, :], m[:, None, :])
    def evolve(self, r0, m0, far):
        r = r0.clone()
        m = m0.clone()
        tetra_idx = self.find_tetrahedron_point(r0)

        i = 0
        r_hist = [r]
        m_hist = [m]
        distances = [torch.zeros(r.shape[0], device=r.device)]
        cumulative_distance = torch.zeros_like(distances[0])
        tetra_hist = [tetra_idx]
        while i < self.n_steps and (tetra_idx != -1).all() and cumulative_distance.min() < far:
            tetra_idx, r, m, d = self.evolve_in_tetrahedron(tetra_idx, r, m)  # , debug_pos=866)
            tetra_idx = self.find_tetrahedron_point(r)
            # assert (tetra_idx_search == tetra_idx).all()
            r_hist.append(r)
            m_hist.append(m)
            distances.append(d)
            tetra_hist.append(tetra_idx)
            cumulative_distance += d
            i += 1
            # assert tetra_idx == self.find_tetrahedron_point(r)
        # print('hola', i)

        r_hist = torch.stack(r_hist, dim=1)
        m_hist = torch.stack(m_hist, dim=1)
        distances = torch.stack(distances, dim=1)
        tetra_hist = torch.stack(tetra_hist, dim=1)
        # from libs.plot_helpers import plot_tetra
        # plot_tetra(self.graph, tetra_hist, r_hist, m_hist, b_pos=581)

        distances = torch.cumsum(distances, dim=1)

        return {'r_hist': r_hist, 'm_hist': m_hist, 'distances': distances, 'tetra_hist': tetra_hist}

    def find_tetrahedron_point(self, point):
        # Very much from https://stackoverflow.com/questions/25179693/how-to-check-whether-the-point-is-in-the-tetrahedron-or-not
        n_p = point.shape[0]
        orir = torch.repeat_interleave(self.graph.origin.unsqueeze(-1), n_p, dim=2)
        newp = torch.einsum('imk, kmj -> kij', self.graph.ort_tetra, point.T - orir)
        val = (torch.all(newp >= 0, dim=1) & torch.all(newp <= 1, dim=1) & (torch.sum(newp, dim=1) <= 1))
        id_tet, id_p = torch.nonzero(val, as_tuple=True)

        res = - torch.ones(n_p, dtype=id_tet.dtype, device=id_tet.device)  # Sentinel value
        res[id_p] = id_tet
        return res


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mesh_type = '3D'
    filename = '../meshes/sphere_fine.msh'
    near = 0
    far = 1.5
    N_samples = 2
    r0 = torch.tensor([[0.44940298795700073, 0.40466320514678955, -0.5139325261116028], [0.4494, 0.40466, -0.51393]],
                      device=device)
    m0 = torch.tensor([[-0.5564, -0.5963, 0.5787]], dtype=torch.float32, device=device)
    m0 = m0 / m0.norm(dim=1)
    tr = [t.TetraToEdge(remove_tetras=False), t.TetraToNeighbors(), t.TetraCoordinates()]

    ev = EvolutionModel(filename, n_steps=10, transforms=tr)
    ev.to(device)
    ev.train()

    tet_batch = ev.find_tetrahedron_point(r0)
    tet0 = ev.find_tetrahedron_point(r0[0].unsqueeze(0))
    tet1 = ev.find_tetrahedron_point(r0[1].unsqueeze(0))
    assert tet_batch[0] == tet0
    assert tet_batch[1] == tet1
    n_index = - 0.5 * (ev.graph.pos.norm(dim=1).max() - ev.graph.pos.norm(dim=1)) + 1.5

    ev.init_vars(n_index)

    f = ev.evolve(r0, m0)

    # crit = nn.MSELoss()
    # real = torch.randn_like(f)
    # loss = crit(f, real)
    # loss.backward()
