import pickle
from tqdm import tqdm
from more_itertools import chunked
from scipy.sparse import coo_matrix

data, row, col = [], [], []
with open("weibo_network.txt") as f:
    for i, l in enumerate(tqdm(f, total=1787444)):
        if i == 0:
            N, M = list(map(int, l.strip().split()))
            print(N, M)
        else:
            v1_id, k, *arr = list(map(int, l.strip().split()))
            assert len(arr) == 2 * k

            # the users "FOLLOWED" by v1, "1" indicates that user v1 and v2 have a reciprocal FOLLOW relationship, while "0" indicates that their relationship is not reciprocal.
            for v2_id, val in chunked(arr, n=2):
                data.append(1)
                row.append(v1_id)
                col.append(v2_id)
                if val == 1:
                    data.append(1)
                    row.append(v2_id)
                    col.append(v1_id)


a = coo_matrix((data,(row,col)), dtype=int).tocsr()
print(a)
with open("weibo.sparse.pl", "wb") as f:
    pickle.dump(a, f)

