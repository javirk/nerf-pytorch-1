import torch
import os
import numpy as np
from torch_geometric.utils import to_undirected
from itertools import combinations


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


class TetraToNeighbors(object):
    """Converts a graph of tetrahedrons to two matrixes: the faces of each tetrahedron (Tx4) and
    the tetrahedrons that share a face (Fx2)"""

    def __init__(self, mesh_folder):
        self.mesh_folder = mesh_folder

    def __call__(self, data):
        files = os.listdir(self.mesh_folder)
        if 'face_tetra.npy' in files and 'face_vertex.npy' in files and 'tetra_face.npy' in files:
            data = self.load_from_disk(data)
        else:
            data = self.generate_variables(data)
            self.save_to_disk(data)

        return data

    def load_from_disk(self, data):
        data.tetra_face = torch.from_numpy(np.load(os.path.join(self.mesh_folder, 'tetra_face.npy')))
        data.face_tetra = torch.from_numpy(np.load(os.path.join(self.mesh_folder, 'face_tetra.npy')))
        data.face_vertex = torch.from_numpy(np.load(os.path.join(self.mesh_folder, 'face_vertex.npy')))
        return data

    @staticmethod
    def generate_variables(data):
        """
        return:
        tetra_face: torch.Tensor(num_tetra, 4). Connects tetrahedrons (idx) to their faces (values)
        face_tetra: torch.Tensor(num_faces, 2). Connects faces (idx) to the tetrahedrons that share them (values).
            Each face must belong to at most 2 tetrahedrons. -1 where there is no tetrahedron i.e. at the edge.
        """
        assert data.tetra is not None, 'The graph must have tetrahedron data'
        N_tetra = data.tetra.shape[1]
        N_edges = data.edge_index.shape[1] // 2  # Divide by 2 because each edge happens twice
        N_faces = 1 + N_tetra + N_edges - data.pos.shape[0]  # V-E+F-C = 1 --> F = 1 + C + E - V

        # tetra_face = torch.tensor(list(range(N_tetra * 4)), dtype=torch.int64).reshape(N_tetra, 4)  # This is absolutely wrong
        tetra_face = - torch.ones((N_tetra, 4), dtype=torch.int64)
        face_tetra = - torch.ones((N_faces, 2), dtype=torch.int64)
        face_vertex = torch.ones((N_faces, 3), dtype=torch.int64)
        vertex_used = []

        i_face = -1
        combs = list(combinations(range(4), 3))
        from tqdm import tqdm
        for i_tetra in tqdm(range(N_tetra)):
            vtx_tetra = data.tetra[:, i_tetra]
            tetra_faces = []
            for i, j, k in combs:
                i_v = vtx_tetra[i]
                j_v = vtx_tetra[j]
                k_v = vtx_tetra[k]

                vtx_tensor, _ = torch.tensor([i_v, j_v, k_v]).sort()
                list_vtx_tensor = vtx_tensor.tolist()

                if list_vtx_tensor not in vertex_used:
                    vertex_used.append(list_vtx_tensor)
                    i_face += 1

                    face_vertex[i_face] = vtx_tensor

                    adj_tetra = torch.where((data.tetra == i_v).any(0) & (data.tetra == j_v).any(0) &
                                            (data.tetra == k_v).any(0))[0]

                    if len(adj_tetra) == 1:
                        face_tetra[i_face, 0] = adj_tetra
                    else:  # 2. Maximum is 2 and 0 is impossible.
                        face_tetra[i_face] = adj_tetra
                    tetra_faces.append(i_face)
                else:
                    tetra_faces.append(vertex_used.index(list_vtx_tensor))

            tetra_face[i_tetra] = torch.tensor(tetra_faces)

        data.tetra_face = tetra_face
        data.face_tetra = face_tetra
        data.face_vertex = face_vertex
        return data

    def save_to_disk(self, data):
        np.save(os.path.join(self.mesh_folder, 'tetra_face.npy'), data.tetra_face.cpu().numpy())
        np.save(os.path.join(self.mesh_folder, 'face_tetra.npy'), data.face_tetra.cpu().numpy())
        np.save(os.path.join(self.mesh_folder, 'face_vertex.npy'), data.face_vertex.cpu().numpy())

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class TetraCoordinates(object):
    r"""
    Given a list of the xyz coordinates of the vertices of a tetrahedron,
    return tetrahedron coordinate system
    """

    def __init__(self):
        pass

    def __call__(self, data):
        num_tetra = data.tetra.shape[1]
        # This can be done in one line, but...
        ori = data.pos[data.tetra[0]]
        v1 = data.pos[data.tetra[1]] - ori
        v2 = data.pos[data.tetra[2]] - ori
        v3 = data.pos[data.tetra[3]] - ori

        v1r = v1.T.reshape((3, 1, num_tetra))
        v2r = v2.T.reshape((3, 1, num_tetra))
        v3r = v3.T.reshape((3, 1, num_tetra))
        # mat defines an affine transform from the tetrahedron to the orthogonal system
        mat = torch.cat((v1r, v2r, v3r), dim=1)
        # The inverse matrix does the opposite (from orthogonal to tetrahedron)
        inv_mat = torch.linalg.inv(mat.to(torch.float64).T).T

        data.ort_tetra = inv_mat.to(torch.float32)
        data.origin = ori
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
