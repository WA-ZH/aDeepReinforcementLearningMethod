
import torch
import numpy as np


def get_random_problems(batch_size, problem_size, n_size, pomo_size, ndir=6, nmac=3, ntool=9):

    half_pomo = int(pomo_size/2)

    stc = np.zeros((batch_size, problem_size, 5), dtype=int)  # Initialize static variables

    for b in range(batch_size):
        stc[b, 0, 0] = 1
        stc[b, 1:n_size, 0] = np.random.permutation(np.arange(2, n_size + 1))

        # Set two connector IDs in the first two columns of stc
        for j in range(half_pomo):
            stc[b, j, 1] = 0

        for j in range(half_pomo, n_size):
            stc[b, j, 1] = np.random.randint(1, stc[b, j, 0].astype(int))

        for j in range(n_size, n_size+half_pomo):
            idx = np.random.randint(half_pomo)
            stc[b, j, :2] = stc[b, idx, :2].astype(int)

        for j in range(n_size + half_pomo, problem_size):
            idx = np.random.randint(half_pomo, n_size)
            stc[b, j, :2] = stc[b, idx, :2].astype(int)


    stc[:, :, 2] = np.random.randint(ndir, size=(batch_size, problem_size))
    stc[:, :, 3] = np.random.randint(nmac, size=(batch_size, problem_size))
    stc[:, :, 4] = np.random.randint(ntool, size=(batch_size, problem_size))

    #stc = stc[stc[:, :, 1].argsort()]

    node_demand = np.ones((batch_size, problem_size))

    static = torch.tensor(stc).float()
    node_demand = torch.tensor(node_demand).float()


    sorted_indices = torch.argsort(static[:, :, 1])

    static = torch.stack([static[i][sorted_indices[i], :] for i in range(stc.shape[0])])



    return static, node_demand



