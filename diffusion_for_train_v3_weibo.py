import argparse
import copy
import pickle
import random
from collections import Counter
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import networkx as nx
import scipy.sparse as sp
import past
from igraph import Graph
import numpy as np
import torch
from tqdm import tqdm, trange

def iteration_bunch(model, bunch_size, node_status=True, progress_bar=False):
    system_status = []
    pre_its = ''
    for it in tqdm(past.builtins.xrange(0, bunch_size), disable=not progress_bar):
        its = model.iteration(node_status)
        if not isinstance(pre_its, str):
            if is_diffusion_stable(pre_its, its):
                print("Diffusion has reached stable state.")
                break
        system_status.append(its)
        pre_its = its
    return system_status

def is_diffusion_stable(prev_its, curr_its):
    # 比较"status"字段是否相同
    if prev_its["status"] != curr_its["status"]:
        return False
    return True

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
    iterations = iteration_bunch(model, iter_num, progress_bar=True)
    end = [model.status[key] for key in sorted(model.status.keys())]
    return start, end

def get_adj(dataset, diffusion_model, seed_rate):
    print('data/' + dataset + '_mean_' + diffusion_model + str(10 * seed_rate) + '.SG')
    with open('data/' + dataset + '_mean_' + diffusion_model + str(10 * seed_rate) + '.SG', 'rb') as f:
        graph_adj = pickle.load(f)
        adj, inverse_pairs = graph_adj['adj'], graph_adj['inverse_pairs']
        return adj

def get_feature_of_graph(adj):
    G = nx.from_scipy_sparse_array(adj)
    #没有边的孤立节点个数
    isolated_nodes = list(nx.isolates(G))
    num_isolated = len(isolated_nodes)
    print(f"Number of isolated nodes (without edges): {num_isolated}")
    #边数最大的节点的边数
    degrees = dict(G.degree())
    max_degree = max(degrees.values())
    print(f"Maximum node degree: {max_degree}")
    #边数最大的节点个数
    max_degree_nodes = [n for n, d in degrees.items() if d == max_degree]
    num_max_degree = len(max_degree_nodes)
    print(f"Number of nodes with maximum degree: {num_max_degree}")
    # 计算每个节点的度数
    degrees = [d for n, d in G.degree()]
    degree_counts = Counter(degrees)
    print("Node Degree Distribution:")
    count_num = 0
    for d, count in sorted(degree_counts.items()):
        if d <= 50:
            count_num += count
        print(f"Degree {d}: {count} nodes")
    print(f"Degree bewteen 1 and 50 nodes number is {count_num - num_isolated}")
def get_top_nodes_and_edge(adj, node_num):
    sub_adj = adj[:node_num, :node_num]
    print("===================================")
    print(sub_adj)
    print("******************************************************")
    get_feature_of_graph(sub_adj)
    print("******************************************************")
    return sub_adj

if __name__ == "__main__":
    print('Is GPU available? {}\n'.format(torch.cuda.is_available()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description="GenIM")
    datasets = ['jazz', 'cora_ml', 'power_grid', 'netscience', 'random5', 'weibo']
    parser.add_argument("-d", "--dataset", default="weibo", type=str,
                        help="one of: {}".format(", ".join(sorted(datasets))))
    diffusion_models = ['IC', 'LT', 'SIS']
    parser.add_argument("-dm", "--diffusion_model", default="IC", type=str,
                        help="one of: {}".format(", ".join(sorted(diffusion_models))))
    seed_rate = [1, 5, 10, 20]
    parser.add_argument("-sp", "--seed_rate", default=1, type=int,
                        help="one of: {}".format(", ".join(str(sorted(seed_rate)))))
    mode = ['Normal', 'Budget Constraint']
    parser.add_argument("-m", "--mode", default="normal", type=str,
                        help="one of: {}".format(", ".join(sorted(mode))))
    parser.add_argument("-it", "--iter_num", default=100000, type=int,
                        help="Simulation steps.")
    args = parser.parse_args()
    print(f"{args.dataset}.sparse.pl")
    with open(f"{args.dataset}.sparse.pl", 'rb') as f:
        adj = pickle.load(f)

    adj = get_top_nodes_and_edge(adj, 50000)

    inverse_pairs = np.zeros((100, adj.shape[0], 2))
    for i in trange(100, desc="Sampling"):
        seeds = random.sample(range(adj.shape[0]), k=int(adj.shape[0] * args.seed_rate * 0.01))
        # print("Sampling {} seeds!".format(len(seeds)))
        x, y = diffusion_inverse_pairs(adj, seeds, args.diffusion_model, args.iter_num)
        inverse_pairs[i] = np.array(list(zip(x, y)))

    inverse_pairs = torch.from_numpy(inverse_pairs).float()
    graph = {
        'adj': adj,
        'inverse_pairs': inverse_pairs,
    }

    with open(args.dataset + '_mean_' + args.diffusion_model + str(10 * args.seed_rate) + '.SG.new', 'wb') as f:
        pickle.dump(graph, f)
