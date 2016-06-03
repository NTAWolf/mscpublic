import numpy as np

edges = {
     1: [3, 15],
     2: [3, 15],
     4: [5],
     5: [2, 6, 17],
     6: [2, 17],
     7: [3, 16],
     8: [3, 16],
     9: [10],
    10: [8, 11, 17],
    11: [8, 17],
    12: [7, 8, 9],
    13: [14],
    14: [1, 2, 4],
    15: [2, 4],
    16: [8, 9],
    21: [23],
    22: [23],
    23: [12, 13],
}

edges_list = [(x,z) for x in edges for z in edges[x]]

descriptions = {
     1: 'reinspection L1',
     2: 'r.i. L1 emergency',
     3: 'r.i. discard',
     4: 'toe cut L1',
     5: 'weight L1',
     6: 'sample track L1',

     7: 'reinspection L2',
     8: 'r.i. L2 emergency',
     9: 'toe cut L2',
    10: 'weight L2',
    11: 'sample track L2',

    12: 'fat end detacher L2',
    13: 'fat end detacher L1',
    14: 'intestine remover L1',
    15: 'reinspection L1 out',
    16: 'reinspection L2 out',
    17: 'cooling tunnel',
    21: 'gambrelling table L1',
    22: 'gambrelling table L2',
    23: 'autofom',
}

positions = {
                            17:(0, 10),

     5:(-2, 8), 6:(-1, 8),              11:(1, 8), 10:(2, 8),

     4:(-2, 7),                                     9:(2,  7),

                 2:(-1, 6),  3:(0, 6),   8:(1,  6),

                15:(-1, 5),             16:(1,  5),

                 1:(-1, 4),              7:(1,  4),

    14:(-2, 3),
    13:(-2, 2),                                    12:(2, 2),
                             23:(0, 1),
    21:(-2, 0),                                    22:(2, 0),
}

nodes = list(edges)

def edges_to_line_segments(edges):
    """edges is a list of (start,end) tx pairs
    Returns an np.array of
        [[start.x, start.y],
         [  end.x,   end.y]]
    """
    out = np.zeros((len(edges),2,2))
    for i, (start,end) in enumerate(edges):
        out[i] = np.array([positions[start], positions[end]])
    return out