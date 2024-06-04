# DeepIM for Weibo

The code has been tested with PyTorch 2.3.0 and Python 3.11.5, simulating and constructing the network, and sampling a subnetwork of 50,000 nodes for training and testing.

Verify the performance impact on a single A100 GPU.

## Prepare Data

Download the weibo_network.txt and place it in the root path of the current project.

## Getting Started

### Step 1

Run `load_to_sparse_array_fast.py`  to load `web_network.txt` and generate `weibo.sparse.pl` in the current path. The `*.sparse.pl` file contains the entire social network data of weibo_network.

### Step 2

Run `diffusion_for_train_weibo.py` to generate a sampled graph of Weibo. The default parameters will load `weibo.sparse.pl`.  

Line **110~118** code in `diffusion_for_train_weibo.py` define the dataset name, diffusion model, and seed rate of the trainable sub-graph. You can change these parameters to load different datasets using various diffusion models and seed rates. In the seed rate parameter, 1 represents 0.01, 5 represents 0.05, 10 represents 0.1, and 20 represents 0.2 relative to the entire graph node. The default parameters will use **weibo dataset, IC diffusion model and 0.01 seed rate** to build a trainable sub-graph.

Line **129** code  `adj = get_top_nodes_and_edge(adj, 50000)`  in `diffusion_for_train_weibo.py` decides the size of  the sampled graph.  `50000`means the code will sample a graph containing 50000 nodes and corresponding edges.

The final product of `diffusion_for_train_weibo.py` is a trainable graph dataset file named `*.SG.new`, the final product called `weibo_mean_IC10.SG.new` if you haven't change any parameters.

### Step 3

Run `diffusion_for_test.py` to train model and verify the impact scope, the result of `diffusion_for_test.py` represents, on average, how many nodes will be affected by one node. Same as step 2,  line 35~43 define the name of file to be read. The default parameters will load a file called `weibo_mean_IC10.SG.new` and print the result on the screen.

## Result of experiment

We use smallest dataset **jazz** and biggest dataset **power_grid** in DeepIM official datasets to conduct a series experiments.

|                     | our result | paper result | official code result |
| :-----------------: | :--------: | :----------: | :------------------: |
| LT diffusion model  |   39.89    |     99.1     |        81.81         |
| IC diffusion model  |   50.65    |     49.9     |        47.82         |
| SIS diffusion model |   71.11    |     74.1     |        69.34         |

The above table is **jazz (seed rate = 20%)** result, the overall results are consistent with offical result expect LT diffusion model, which has the lowest result. So we use **power_grid (seed rate = 20%)** dataset and focus on IC and SIS diffusion model to conduct further experiments.

|                     | our result | paper result | official code result |
| :-----------------: | :--------: | :----------: | :------------------: |
| IC diffusion model  |   53.73    |     52.4     |        50.38         |
| SIS diffusion model |   22.04    |     23.8     |        23.81         |

The above table is **power_grid (seed rate = 20%)** result, the result are still consistent with the official result, so we conduct further experiments on weibo network. We sample two sub-graph by obtaining the first **N** nodes in original 178w+ weibo graph ourselves, **N** is 10000 and 50000 respectively.

|                     | weibo(seed rate=20%)<br>1w node | weibo(seed rate=1%)<br>1w node | weibo(seed rate=1%)<br>5w node |
| :-----------------: | :---------------------------: | :--------------------------: | :--------------------------: |
| LT diffusion model  |             56.1              |              /               |              /               |
| IC diffusion model  |             37.25             |            14.91             |            36.24             |
| SIS diffusion model |             28.14             |              /               |              /               |

