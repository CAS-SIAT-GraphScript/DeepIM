import argparse
import copy
import pickle
import random

import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import networkx as nx
import past
from igraph import Graph
import numpy as np
import torch
from tqdm import tqdm, trange


def diffusion_inverse_pairs(adj_matrix, seed_nodes, diffusion='LT', iter_num=100):
    G = nx.from_scipy_sparse_array(adj_matrix)
    # G = Graph.Adjacency(adj_matrix)
    if diffusion == 'LT':
        model = ep.ThresholdModel(G)
        config = mc.Configuration()
        for n in G.nodes():
            config.add_node_configuration("threshold", n, 0.4)
    elif diffusion == 'IC':
        model = ep.IndependentCascadesModel(G)
        config = mc.Configuration()
        for e in G.edges():
            config.add_edge_configuration("threshold", e, 1 / nx.degree(G)[e[1]])
    elif diffusion == 'SIS':
        model = ep.SISModel(G)
        config = mc.Configuration()
        config.add_model_parameter('beta', 0.001)
        config.add_model_parameter('lambda', 0.001)
    else:
        raise ValueError('Only IC, LT and SIS are supported.')

    config.add_model_initial_configuration("Infected", seed_nodes)

    model.set_initial_status(config)

    start = [model.status[key] for key in sorted(model.status.keys())]
    iterations = model.iteration_bunch(iter_num, progress_bar=True)
    end = [model.status[key] for key in sorted(model.status.keys())]
    return start, end

def get_adj(dataset, diffusion_model, seed_rate):
    print('data/' + dataset + '_mean_' + diffusion_model + str(10 * seed_rate) + '.SG')
    with open('data/' + dataset + '_mean_' + diffusion_model + str(10 * seed_rate) + '.SG', 'rb') as f:
        graph_adj = pickle.load(f)
        adj, inverse_pairs = graph_adj['adj'], graph_adj['inverse_pairs']
        return adj

if __name__ == "__main__":
    print('Is GPU available? {}\n'.format(torch.cuda.is_available()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description="GenIM")
    datasets = ['jazz', 'cora_ml', 'power_grid', 'netscience', 'random5', 'weibo']
    parser.add_argument("-d", "--dataset", default="power_grid", type=str,
                        help="one of: {}".format(", ".join(sorted(datasets))))
    diffusion_models = ['IC', 'LT', 'SIS']
    parser.add_argument("-dm", "--diffusion_model", default="SIS", type=str,
                        help="one of: {}".format(", ".join(sorted(diffusion_models))))
    seed_rate = [1, 5, 10, 20]
    parser.add_argument("-sp", "--seed_rate", default=20, type=int,
                        help="one of: {}".format(", ".join(str(sorted(seed_rate)))))
    mode = ['Normal', 'Budget Constraint']
    parser.add_argument("-m", "--mode", default="normal", type=str,
                        help="one of: {}".format(", ".join(sorted(mode))))
    args = parser.parse_args()

    # with open(f"{args.dataset}.sparse.pl", 'rb') as f:
    #     adj = pickle.load(f)

    adj = get_adj(args.dataset, args.diffusion_model, args.seed_rate)

    inverse_pairs = np.zeros((100, adj.shape[0], 2))
    for i in trange(100, desc="Sampling"):
        seeds = random.sample(range(adj.shape[0]), k=int(adj.shape[0] * args.seed_rate * 0.01))
        # print("Sampling {} seeds!".format(len(seeds)))
        x, y = diffusion_inverse_pairs(adj, seeds, args.diffusion_model)
        inverse_pairs[i] = np.array(list(zip(x, y)))

    inverse_pairs = torch.from_numpy(inverse_pairs).float()
    graph = {
        'adj': adj,
        'inverse_pairs': inverse_pairs,
    }

    with open(args.dataset + '_mean_' + args.diffusion_model + str(10 * args.seed_rate) + '.SG.new', 'wb') as f:
        pickle.dump(graph, f)
