import meshio
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from einops import rearrange


class TetraToEdge(object):
    r"""Converts mesh tetras :obj:`[4, num_tetras]` to edge indices
    :obj:`[2, num_edges]`.
    Args:
        remove_tetras (bool, optional): If set to :obj:`False`, the tetra tensor
            will not be removed.
    """

    def __init__(self, remove_tetras=True):
        self.remove_tetras = remove_tetras

    def __call__(self, data):
        if data.tetra is not None:
            tetra = data.tetra
            edge_index = torch.cat([tetra[:2], tetra[1:3, :], tetra[-2:], tetra[::2], tetra[::3], tetra[1::2]], dim=1)
            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

            data.edge_index = edge_index
            if self.remove_tetras:
                data.tetra = None

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


def read_mesh(mesh_path):
    mesh = meshio.read(
        mesh_path,  # string, os.PathLike, or a buffer/open file
    )
    data = from_meshio(mesh)
    data = TetraToEdge(remove_tetras=False)(data)
    return data


def from_meshio(mesh):
    r"""Converts a :.msh file to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        mesh (meshio.read): A :obj:`meshio` mesh.
    """

    if meshio is None:
        raise ImportError('Package `meshio` could not be found.')

    pos = torch.from_numpy(mesh.points).to(torch.float)
    tetra = torch.from_numpy(mesh.cells_dict['tetra']).to(torch.long).t().contiguous()
    face = torch.from_numpy(mesh.cells_dict['triangle']).to(torch.long).t().contiguous()
    return Data(pos=pos, tetra=tetra, face=face)


def transfer_batch_to_device(batch, device):
    for k, v in batch:
        if hasattr(v, 'to'):
            batch[k] = v.to(device)
    return batch

def smart_sort(x, permutation):
    """
    Sorting tensor x along the first dimension based on permutation
    :param x: torch.Tensor
    :param permutation: torch.Tensor
    :return:
    """
    d1, d2, d3 = x.size()
    ret = x[permutation.flatten(),
        torch.arange(d2).unsqueeze(0).repeat((d1, 1)).flatten()
    ].view(d1, d2, d3)
    return ret

"""def smart_sort_old(x, permutation):
    d1, d2, d3 = x.size()
    ret = x[
        torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
        permutation.flatten()
    ].view(d1, d2, d3)
    return ret"""

def get_pts_zvals(rays_o, rays_d, graph, N_samples, sampling_probability, perturb_points=False):
    """
    Given a graph and a set of rays, compute the distance to each node of the graph and sample random nodes.
    TODO: Use the near and far parameters.
    :param rays_o: N_rays, 3
    :param rays_d: N_rays, 3
    :param graph:
    :param N_samples:
    :param sampling_probability: float. This comes from the config file.
    :param perturb_points:
    :return: pts [N_rays, N_samples, 3] and z_vals [N_rays, N_samples]
    """
    ## This distance and probability will change later, this will be predicted by the GCN. Now it's theoretical
    distn = torch.norm(torch.cross(rays_o - graph.pos.unsqueeze(1), rays_d.repeat(graph.num_nodes, 1, 1), dim=-1),
                       dim=-1) / torch.norm(rays_d, dim=-1)  # num_nodes x rays
    distn = distn / distn.max(dim=0).values  # Normalize
    probs = 1 - distn
    ## Until here

    dist_b = (probs > sampling_probability).float()
    idx = torch.multinomial(dist_b.permute(1, 0), N_samples)
    pts = graph.pos[idx]

    if perturb_points > 0:
        pts += torch.randn_like(pts)

    pts = rearrange(pts, 'r s d -> s r d')
    z_vals = torch.norm(pts - rays_o, dim=-1)
    z_vals, i_permutation = torch.sort(z_vals, dim=0)
    pts = smart_sort(pts, permutation=i_permutation)

    # TODO: This is awful, maybe there is a way to not change so many times. Everything comes from pts_d - rays_o
    pts = rearrange(pts, 's r d -> r s d')
    z_vals = rearrange(z_vals, 's r -> r s')

    return pts, z_vals