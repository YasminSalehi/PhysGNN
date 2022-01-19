from preprocessing import *

'''
Note: After creating the adjacency mattrix file for one dataset, the same
file can be used for the second dataset as the mesh remains the same across
both datasets. Also, please note that only 450 graph data elements (pytorch geometric data elements)
can be generated in google colab, which is the number of directions 15 and time steps 30
multiplied by each other. 


For Dataset 1, we have 30 time steps, 15 directions, and 11 nodes.
Therefore, you create the A.csv file once, and you use it 11 times
for dataset_1/a, dataset_1/b, ..., and  dataset_1/k files, where dataset_1/a
holds information about applying forces to node 1, dataset_1/b
holds information about applying forces to node 2, ..., and dataset_1/k
holds information about applying forces to node 11.


For Dataset 2, we have 30 time steps and 165 directions, where we apply forces 
to 100 prescribed load nodes. Therefore, you create the A.csv file once, and you 
use it 11 times for dataset_2/a, dataset_2/b, ..., and dataset_2/k files, where 
dataset_2/a holds information about applying forces to the prescribed load nodes
for directions 1 - 15, dataset_2/b holds information about applying forces to the 
prescribed load nodes for directions 16 - 30, ..., and dataset_2/k holds information 
about applying forces to the  prescribed load nodes for directions 151 - 165.

'''


def dataset_1_adj_matrix_gen():
    input_path = 'Data_Generator/code/input_1/'

    # Initialization
    elements_filename = input_path + 'elements.csv'

    num_nodes = 9118  # number of nodes per each graph
    num_dirs = 15
    t_steps = 30


    formatted_data_path = 'dataset_1/'
    adj_matrix_filename = formatted_data_path + 'A.csv'

    A = adj_matrix_builder(elements_filename, num_nodes)
    adj_matrix_full_format(A, num_nodes, num_dirs, t_steps, adj_matrix_filename)


def dataset_2_adj_matrix_gen():
    input_path = 'Data_Generator/code/input_2/'

    # Initialization
    elements_filename = input_path + 'elements.csv'

    num_nodes = 9118  # number of nodes per each graph
    num_dirs = 15
    t_steps = 30

    formatted_data_path = 'dataset_2/'
    adj_matrix_filename = formatted_data_path + 'A.csv'

    A = adj_matrix_builder(elements_filename, num_nodes)
    adj_matrix_full_format(A, num_nodes, num_dirs, t_steps, adj_matrix_filename)


