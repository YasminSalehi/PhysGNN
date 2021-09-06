# Importing what we need
from library_imports import *


def Graph_load_batch(name, min_num_nodes=20, max_num_nodes=9118):
    """
    load many graphs
    :return: a list of graphs
    """
    print('Loading graph dataset: ' + str(name))

    G = nx.Graph()

    # load data
    # ---------------------------------------------------------------------------------------
    print('Loading the adjacency matrix...')
    data_adj = np.loadtxt(name + '/A.csv', delimiter=',').astype(int)
    # ---------------------------------------------------------------------------------------
    print('Loading the graph indicator...')
    data_graph_indicator = np.loadtxt(path + 'graph_indicator.csv', delimiter=',').astype(int)
    # ---------------------------------------------------------------------------------------
    print('Loading the graph labels...')
    data_graph_labels = np.loadtxt(path + 'graph_labels.csv', delimiter=',').astype(int)
    # ---------------------------------------------------------------------------------------
    print('Loading the node attributes...')
    data_node_att = np.loadtxt(path + 'node_attributes_raw.csv', delimiter=',')
    # ---------------------------------------------------------------------------------------
    print('Loading the node labels...')
    node_labels = np.loadtxt(path + 'output_displacement.csv', delimiter=',')
    # ---------------------------------------------------------------------------------------
    # #######################################################################################
    print('Data loaded.')
    # #######################################################################################
    # ---------------------------------------------------------------------------------------
    print('Generating data tuples...')
    data_tuple = list(map(tuple, data_adj))
    # ---------------------------------------------------------------------------------------
    print('Adding edges...')
    G.add_edges_from(data_tuple)
    # ---------------------------------------------------------------------------------------
    print('Adding features and node labels to graph nodes...')
    for i in range(node_labels.shape[0]):
        G.add_node(i + 1, feature=data_node_att[i])
        G.add_node(i + 1, label=node_labels[i])
    # ---------------------------------------------------------------------------------------
    print('Removing isolated nodes...')
    print(list(nx.isolates(G)))
    G.remove_nodes_from(list(nx.isolates(G)))
    # ---------------------------------------------------------------------------------------
    print('Splitting data into graphs...')
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        G_sub.graph['label'] = data_graph_labels[i]
        if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes() <= max_num_nodes:
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
    # ---------------------------------------------------------------------------------------
    print('Loaded')
    return graphs


def nx_to_tg_data(graphs):
    data_list = []

    for i in range(len(graphs)):

        graph = graphs[i].copy()

        # relabel graphs
        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))
        nx.relabel_nodes(graph, mapping, copy=False)

        # ----------------------------------------------------------------------

        feature_values = nx.get_node_attributes(graph, 'feature').values()
        label_values = nx.get_node_attributes(graph, 'label').values()
        num_nodes = len(feature_values)

        # ----------------------------------------------------------------------
        # Node attribute format
        # {x} {y} {z} {physical_property} {F_x} {F_y} {F_z}
        j = 0
        features = []
        for value in feature_values:
            if j == 0:
                features = value
            else:
                features = np.concatenate((features, value), axis=0)
            j += 1
        # ----------------------------------------------------------------------
        # Node label format
        # {d_x} {d_y} {d_z}
        j = 0
        node_labels = []
        for value in label_values:
            if j == 0:
                node_labels = value
            else:
                node_labels = np.concatenate((node_labels, value), axis=0)
            j += 1
        # ----------------------------------------------------------------------
        features = features.reshape((num_nodes, 7))  # because we have 7 features
        features = torch.from_numpy(features).float()

        node_labels = node_labels.reshape((num_nodes, 3))  # because we have 3 outputs
        node_labels = torch.from_numpy(node_labels).float()

        pos = features[:, 0:3]
        x = features[:, 3:]

        # ----------------------------------------------------------------------
        # get edges
        edge_index = np.array(list(graph.edges))
        edge_index = np.concatenate((edge_index, edge_index[:, ::-1]), axis=0)
        edge_index = torch.from_numpy(edge_index).long().permute(1, 0)

        print(edge_index)

        # get edge_labels
        edge_values = np.ones((edge_index.shape[1], 1))
        edge_values = torch.tensor(edge_values).float()

        for n in range(edge_index.shape[1]):
            node_1 = edge_index[0, n]
            node_2 = edge_index[1, n]

            x_1 = pos[node_1, 0]
            y_1 = pos[node_1, 1]
            z_1 = pos[node_1, 2]

            x_2 = pos[node_2, 0]
            y_2 = pos[node_2, 1]
            z_2 = pos[node_2, 2]

            edge_values[n, 0] = torch.pow(
                (torch.pow((x_1 - x_2), 2) + torch.pow((y_1 - y_2), 2) + torch.pow((z_1 - z_2), 2)), 0.5)

        edge_values = torch.tensor(edge_values).float()

        # ----------------------------------------------------------------------
        # Checking for dimensionality correctness
        print(pos.shape)
        print(pos)
        print(node_labels.shape)
        print(node_labels)
        print(x.shape)
        print(x)
        # ----------------------------------------------------------------------
        # create the data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_values, pos=pos, y=node_labels)

        data_list.append(data)

        # Progress update
        print(str(i + 1) + '/' + str(len(graphs)) + ' data objects created.')

    return data_list


# ----------------------------------------------------------------------------------------------------------------------
# To create pytorch geometric dataset, select the dataset_name,

dataset_name = 'dataset_1'
# dataset_name = 'dataset_2/'

path = dataset_name + '/a/' #
# path = dataset_name + '/b/' #
# path = dataset_name + '/c/' #
# path = dataset_name + '/d/' #
# path = dataset_name + '/e/' #
# path = dataset_name + '/f/' #
# path = dataset_name + '/g/' #
# path = dataset_name + '/h/' #
# path = dataset_name + '/i/' #
# path = dataset_name + '/j/' #
# path = dataset_name + '/k/' #

graphs_all = Graph_load_batch(dataset_name)
dataset = nx_to_tg_data(graphs_all)
print('Pytorch Geometric dataset has been created.')

path_save = dataset_name + '_pickle/' + dataset_name + '_a_raw.pickle' #
# path_save = dataset_name + '_pickle/' + dataset_name + '_b_raw.pickle' #
# path_save = dataset_name + '_pickle/' + dataset_name + '_c_raw.pickle' #
# path_save = dataset_name + '_pickle/' + dataset_name + '_d_raw.pickle' #
# path_save = dataset_name + '_pickle/' + dataset_name + '_e_raw.pickle' #
# path_save = dataset_name + '_pickle/' + dataset_name + '_f_raw.pickle' #
# path_save = dataset_name + '_pickle/' + dataset_name + '_g_raw.pickle' #
# path_save = dataset_name + '_pickle/' + dataset_name + '_h_raw.pickle' #
# path_save = dataset_name + '_pickle/' + dataset_name + '_i_raw.pickle' #
# path_save = dataset_name + '_pickle/' + dataset_name + '_j_raw.pickle' #
# path_save = dataset_name + '_pickle/' + dataset_name + '_k_raw.pickle' #

torch.save(dataset, path_save)

