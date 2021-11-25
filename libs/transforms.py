import torch
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

    def __init__(self):
        pass

    def __call__(self, data):
        """
        return:
        tetra_face: torch.Tensor(num_tetra, 4). Connects tetrahedrons (idx) to their faces (values)
        face_tetra: torch.Tensor(num_faces, 2). Connects faces (idx) to the tetrahedrons that share them (values).
            Each face must belong to at most 2 tetrahedrons. -1 where there is no tetrahedron i.e. at the edge.
        """
        assert data.tetra is not None, 'The graph must have tetrahedron data'
        N_tetra = data.tetra.shape[1]
        tetra_face = torch.tensor(list(range(N_tetra * 4)), dtype=torch.int64).reshape(N_tetra, 4)
        face_tetra = - torch.ones((4 * N_tetra, 2), dtype=torch.int64)  # All of them are repeated twice. That's because
        # we pass through each face twice (two tetrahedrons share one face, and we iterate in tetrahedrons)
        face_vertex = torch.ones((4 * N_tetra, 3), dtype=torch.int64)  # Many of them are repeated!

        i_face = 0
        combs = list(combinations(range(4), 3))
        for i_tetra in range(N_tetra):
            vtx_tetra = data.tetra[:, i_tetra]
            for i, j, k in combs:
                i_v = vtx_tetra[i]
                j_v = vtx_tetra[j]
                k_v = vtx_tetra[k]
                adj_tetra = torch.where((data.tetra == i_v).any(0) & (data.tetra == j_v).any(0) &
                                        (data.tetra == k_v).any(0))[0]

                if len(adj_tetra) == 1:
                    face_tetra[i_face, 0] = adj_tetra
                else:  # 2. Maximum is 2 and 0 is impossible.
                    face_tetra[i_face] = adj_tetra
                face_vertex[i_face] = torch.tensor([i_v, j_v, k_v])

                i_face += 1

        data.tetra_face = tetra_face
        data.face_tetra = face_tetra
        data.face_vertex = face_vertex
        return data

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
        inv_mat = torch.linalg.inv(mat.T).T

        data.ort_tetra = inv_mat
        data.origin = ori
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
