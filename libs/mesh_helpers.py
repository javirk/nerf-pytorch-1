import glob

import meshio
import torch
import numpy as np
from torch_geometric.data import Data

wrap_to_2pi = lambda x: x % (2 * np.pi) #+ (2 * np.pi) * (x == 0)


def from_meshio(mesh, mesh_type='2D'):
    r"""Converts a :.msh file to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        mesh (meshio.read): A :obj:`meshio` mesh.
    """

    if meshio is None:
        raise ImportError('Package `meshio` could not be found.')

    pos = torch.from_numpy(mesh.points).to(torch.float)
    if mesh_type == '3D':
        tetra = torch.from_numpy(mesh.cells_dict['tetra']).to(torch.long).t().contiguous()
        # face = torch.from_numpy(mesh.cells_dict['triangle']).to(torch.long).t().contiguous()
        return Data(pos=pos, tetra=tetra)
    elif mesh_type == '2D':
        face = torch.from_numpy(mesh.cells_dict['triangle']).to(torch.long).t().contiguous()
        return Data(pos=pos, face=face)


def to_meshio(graph):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`.msh format`.

    Args:
        graph (torch_geometric.data.Data): The data object.
    """

    if meshio is None:
        raise ImportError('Package `meshio` could not be found.')

    points = graph.pos.detach().cpu().numpy()
    tetra = graph.tetra.detach().t().cpu().numpy()

    cells = [("tetra", tetra)]

    return meshio.Mesh(points, cells)


def read_mesh(mesh_folder, dimensions, transforms):
    mesh_path = glob.glob(f'{mesh_folder}/*.msh')
    assert len(mesh_path) == 1, 'More than one mesh file in the folder. Exiting'
    mesh_path = mesh_path[0]
    mesh = meshio.read(
        mesh_path,  # string, os.PathLike, or a buffer/open file
    )
    graph = from_meshio(mesh, mesh_type=dimensions)
    for t in transforms:
        graph = t(graph)
    return graph


def get_neighbour_tetra(graph, current_tetra, exit_face_local):
    exit_face = graph.tetra_face[current_tetra, exit_face_local]  # Face number 2 in the local space of the tetrahedron
    return graph.face_tetra[exit_face]


def batch_dot(v1, v2):
    return torch.einsum('b d, b d -> b', v1, v2)


def transfer_batch_to_device(batch, device):
    for k, v in batch:
        if hasattr(v, 'to'):
            batch[k] = v.to(device)
    return batch


def gather_batch(input, index):
    b, t, p = index.shape
    _, _, c = input.shape
    output = torch.zeros((b, t, p, c), device=input.device)

    for i, (val, idx) in enumerate(zip(input, index)):
        output[i] = val[idx]

    return output
