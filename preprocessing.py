import scipy.io
import numpy as np
import pandas as pd
import csv
from csv import reader
import itertools
from numpy import genfromtxt


def adj_matrix_builder(elements, num_nodes):
    '''

    :param elements: a .csv file which contains the elements of the FE model.
    :param num_nodes: number of nodes.
    :return: the adjacency matrix.
    '''

    A = np.zeros((num_nodes, num_nodes))

    with open(elements, 'r') as read_obj:   # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)       # Iterate over each row in the csv using reader object
        for row in csv_reader:              # row variable is a list that represents a row in csv

            node_1 = int(row[0]) - 1
            node_2 = int(row[1]) - 1
            node_3 = int(row[2]) - 1
            node_4 = int(row[3]) - 1

            A[node_1, node_2] = 1
            A[node_2, node_1] = 1

            A[node_1, node_3] = 1
            A[node_3, node_1] = 1

            A[node_1, node_4] = 1
            A[node_4, node_1] = 1

            A[node_2, node_3] = 1
            A[node_3, node_2] = 1

            A[node_2, node_4] = 1
            A[node_4, node_2] = 1

            A[node_3, node_4] = 1
            A[node_4, node_3] = 1

    return A

# --------------------------------------------------------------------------------------------------

def adj_matrix_format(A, filename):
    '''

    :param A: an adjacency matrix
    :param filename: the name of the file where the formatted adjacency matrix will be stored.
    :return: -
    '''

    num_rows = np.size(A,0)
    num_cols = np.size(A,1)

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for i in range(num_rows):

            for j in range(num_cols):
                list = []
                if A[i,j]!=0:
                    list.append(j+1)
                    list.append(i+1)
                    writer.writerow(list)

# --------------------------------------------------------------------------------------------------

def adj_matrix_full_format(A, num_nodes, num_dir, t_steps, filename):
    '''

    :param A: an adjacency matrix
    :param filename: the name of the file where the formatted adjacency matrix will be stored.
    :return: -
    '''

    num_rows = np.size(A,0)
    num_cols = np.size(A,1)
    counter = 0

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for k in range(num_dir*t_steps):

            for i in range(num_rows):

                for j in range(num_cols):
                    list = []
                    if A[i,j]!=0:
                        list.append(j+1 + (k * num_nodes))
                        list.append(i+1 + (k * num_nodes))
                        writer.writerow(list)
                        counter = counter + 1
                        print(counter)

# --------------------------------------------------------------------------------------------------

def xyz(filename_xyz):
    '''

    :param filename_xyz: a .csv file containing the x y and z coordinates of the nodes within the FE model
    :return: a panda data frame holding the x y z coordinate information.
    '''

    data = pd.read_csv(filename_xyz, header=None)
    df = pd.DataFrame(data)
    df.columns = ['x', 'y', 'z']

    return df

# --------------------------------------------------------------------------------------------------

def node_material_assign(elements, elements_ID, face_boundary, boundary_marker, num_nodes, filename):
    '''

    :param elements: csv file which contains the elements of the FE model.
    :param elements_ID: a .csv file which assigns every element to a material.
    :param num_nodes: number of nodes within the FE model.
    :param filename: a filename where every NODE will be assigned a material property (as opposed to element).
    :return: a panda data frame holding node material information.
    '''

    node_ID = np.zeros((num_nodes, 1), dtype=int)

    with open(elements, 'r') as read_obj_1:
        with open(elements_ID, 'r') as read_obj_2:

            csv_reader_1 = reader(read_obj_1) # elements
            csv_reader_2 = reader(read_obj_2) # element id

            for row_1, row_2 in itertools.zip_longest(csv_reader_1, csv_reader_2):

                val = int(row_2[0])

                for n in row_1:
                    n = int(n)-1
                    node_ID[n, 0] = val

    with open(face_boundary, 'r') as read_obj_3:
        with open(boundary_marker, 'r') as read_obj_4:

            csv_reader_3 = reader(read_obj_3)  # face boundaries
            csv_reader_4 = reader(read_obj_4)  # boundary id

            for row_3, row_4 in itertools.zip_longest(csv_reader_3, csv_reader_4):

                val_b = int(row_4[0])
                print(val_b)

                for n in row_3:
                    n = int(n) - 1
                    node_ID[n, 0] = val_b

    np.savetxt(filename, node_ID, fmt='%i')

    df = pd.DataFrame(list(node_ID), columns=['Mat_ID'])
    print(df)
    return df

# --------------------------------------------------------------------------------------------------

def node_support_assign(support_nodes, num_nodes, filename):
    '''

    :param support_nodes: a .csv file which indicates the node ID of rigid nodes in the FE model.
    :param num_nodes: number of nodes within the FE model.
    :param filename: the place to store a column of size( num_nodes x 1) where rigid nodes have a value of 1.
    :return: a panda data frame marking rigid nodes.
    '''

    support_list = np.zeros((num_nodes, 1), dtype=np.int)

    with open(support_nodes, 'r') as read_obj:

        csv_reader = reader(read_obj)
        for row in csv_reader:
            n = int(row[0])-1
            support_list[n,0]=1

    np.savetxt(filename, support_list, fmt='%i')
    df = pd.DataFrame(list(support_list), columns=['Rigid_ID'])
    return df

# --------------------------------------------------------------------------------------------------

"""
The format of the feature matrix is:

{Node ID} {x} {y} {z} {material_ID} {Rigid} {Force magnitude} {F_x} {F_y} {F_z}
"""
#
def feature_constructor(num_nodes, df_xyz, df_mat_id, df_rigid_id, filename_pres_nodes, filename_force_dir, magnitude, t_steps):
    '''

    :param num_nodes: Number of nodes within the FE mesh
    :param df_xyz: x y z data frame
    :param df_mat_id:  material id data frame
    :param df_rigid_id: support nodes data frame
    :param filename_pres_nodes:  prescribed load nodes csv file
    :param filename_force_dir:  direction of forces (normal vectors) csv file
    :param magnitude: magnitude of the force
    :param t_steps: number of time steps in the FEA
    :return: features data frame
    '''

    pres_nodes = genfromtxt(filename_pres_nodes, delimiter=',', dtype=int)
    force_dirs = genfromtxt(filename_force_dir, delimiter=',')

    idx_repeat = int(len(force_dirs) / len(pres_nodes))
    pres_nodes_full = []

    for p_node in pres_nodes:
        for j in range(idx_repeat):
            pres_nodes_full.append(int(p_node))

    # print(pres_nodes_full)

    df_f_1 = pd.concat([df_xyz, df_mat_id, df_rigid_id], axis=1)
    df_f_copy = pd.concat([df_xyz, df_mat_id, df_rigid_id], axis=1)

    for i in range(len(pres_nodes_full) * t_steps - 1):
        frame = [df_f_1,df_f_copy]
        df_f_1 = pd.concat(frame)
        df_f_1.reset_index(drop=True, inplace=True)

    step_mag = magnitude/t_steps

    force = []
    pres_nodes_idx = []
    m = 0

    for i in range(len(pres_nodes_full)):

        p_node_force_direction = force_dirs[i, :]

        for j in range(t_steps):

            updated_mag = step_mag * (j+1)

            force.append(updated_mag)
            force.append(p_node_force_direction[0])
            force.append(p_node_force_direction[1])
            force.append(p_node_force_direction[2])

            # ----------------------------------------

            idx = (pres_nodes_full[i] - 1) + (m * num_nodes)
            pres_nodes_idx.append(idx)
            m = m + 1

    force = np.reshape(force, (-1, 4)) # columns of this matrix are: magnitude, F_x, F_y, F_z

    # print(force)


    force_length = t_steps * len(pres_nodes_full) * num_nodes


    M = np.zeros((force_length, 4))

    # print(pres_nodes_idx)

    for k in range(len(pres_nodes_idx)):

        # print('------------')
        # print(pres_nodes_idx[k])
        # print(force[k, 0])
        # print(force[k, 1])
        # print(force[k, 2])
        # print(force[k, 3])
        # print('------------')

        M[pres_nodes_idx[k], 0] = force[k, 0]
        M[pres_nodes_idx[k], 1] = force[k, 1]
        M[pres_nodes_idx[k], 2] = force[k, 2]
        M[pres_nodes_idx[k], 3] = force[k, 3]

    # print('Printing M')
    # print(M)

    df_f_2 = pd.DataFrame(data=M, columns=["Magnitude", "F_x", "F_y", "F_z"])

    # print(df_f_1)
    # print(df_f_2)


    frame_2 = [df_f_1, df_f_2]

    df_features = pd.concat(frame_2, axis=1)

    # print(df_features)

    return df_features

# --------------------------------------------------------------------------------------------------

def feature_constructor_2(num_nodes, df_xyz, df_mat_id, df_rigid_id, filename_pres_nodes,
                          filename_force_dir_x, filename_force_dir_y, filename_force_dir_z, magnitude, t_steps):
    '''
    This function is used for when we want to construct the feature matrix when we are applying prescribe forces to
    multiple nodes at the same time.

    :param num_nodes: Number of nodes within the FE mesh
    :param df_xyz: x y z data frame
    :param df_mat_id:  material id data frame
    :param df_rigid_id: support nodes data frame
    :param filename_pres_nodes:  prescribed load nodes csv file
    :param filename_force_dir:  direction of forces (normal vectors) csv file
    :param magnitude: magnitude of the force
    :param t_steps: number of time steps in the FEA
    :return: features data frame
    '''

    pres_nodes = genfromtxt(filename_pres_nodes, delimiter=',', dtype=int)
    force_dir_x = genfromtxt(filename_force_dir_x, delimiter=',')
    force_dir_y = genfromtxt(filename_force_dir_y, delimiter=',')
    force_dir_z = genfromtxt(filename_force_dir_z, delimiter=',')

    idx_repeat = force_dir_x.shape[1]
    print(idx_repeat)

    df_f_1 = pd.concat([df_xyz, df_mat_id, df_rigid_id], axis=1)
    df_f_copy = pd.concat([df_xyz, df_mat_id, df_rigid_id], axis=1)

    for i in range(idx_repeat * t_steps - 1):
        frame = [df_f_1,df_f_copy]
        df_f_1 = pd.concat(frame)
        df_f_1.reset_index(drop=True, inplace=True)

    step_mag = magnitude/(t_steps * len(pres_nodes))

    force_dir_x_full = []
    force_dir_y_full = []
    force_dir_z_full = []
    force_magnitude = []
    pres_nodes_idx = []
    m = 0


    for i in range(idx_repeat):

        for j in range(t_steps):

            for k in range(len(pres_nodes)):

                force_dir_x_full.append(force_dir_x[k, i])
                force_dir_y_full.append(force_dir_y[k, i])
                force_dir_z_full.append(force_dir_z[k, i])

                updated_mag = step_mag * (j + 1)
                force_magnitude.append(updated_mag)

                idx = int(pres_nodes[k]) - 1 + (num_nodes*m)
                pres_nodes_idx.append(idx)

            m = m + 1

    force_length = t_steps * idx_repeat * num_nodes

    M = np.zeros((force_length, 4))

    # print(force_dir_x_full)
    # print(force_dir_y_full)
    # print(force_dir_z_full)
    # print(pres_nodes_idx)

    for k in range(len(pres_nodes_idx)):

        # print('------------')
        # print(pres_nodes_idx[k])
        # print(force[k, 0])
        # print(force[k, 1])
        # print(force[k, 2])
        # print(force[k, 3])
        # print('------------')

        M[pres_nodes_idx[k], 0] = force_magnitude[k]
        M[pres_nodes_idx[k], 1] = force_dir_x_full[k]
        M[pres_nodes_idx[k], 2] = force_dir_y_full[k]
        M[pres_nodes_idx[k], 3] = force_dir_z_full[k]

    # print('Printing M')
    # print(M)

    df_f_2 = pd.DataFrame(data=M, columns=["Magnitude", "F_x", "F_y", "F_z"])

    # print(df_f_1)
    # print(df_f_2)


    frame_2 = [df_f_1, df_f_2]

    df_features = pd.concat(frame_2, axis=1)

    print(df_features)

    return df_features

# --------------------------------------------------------------------------------------------------

def feature_constructor3(num_nodes, df_xyz, df_mat_id, df_rigid_id, filename_pres_nodes, node_idx, filename_force_dir, magnitude, t_steps):
    '''

    :param num_nodes: Number of nodes within the FE mesh
    :param df_xyz: x y z data frame
    :param df_mat_id:  material id data frame
    :param df_rigid_id: support nodes data frame
    :param filename_pres_nodes:  prescribed load nodes csv file
    :param filename_force_dir:  direction of forces (normal vectors) csv file
    :param magnitude: magnitude of the force
    :param t_steps: number of time steps in the FEA
    :return: features data frame
    '''

    pres_nodes = genfromtxt(filename_pres_nodes, delimiter=',', dtype=int)
    print(pres_nodes)
    p_node = pres_nodes[node_idx-1]
    print(p_node)
    force_dirs = genfromtxt(filename_force_dir, delimiter=',')

    idx_repeat = len(force_dirs)
    pres_nodes_full = []


    for j in range(idx_repeat):
        pres_nodes_full.append(int(p_node))

    print(pres_nodes_full)

    df_f_1 = pd.concat([df_xyz, df_mat_id, df_rigid_id], axis=1)
    df_f_copy = pd.concat([df_xyz, df_mat_id, df_rigid_id], axis=1)

    for i in range(len(pres_nodes_full) * t_steps - 1):
        frame = [df_f_1,df_f_copy]
        df_f_1 = pd.concat(frame)
        df_f_1.reset_index(drop=True, inplace=True)

    step_mag = magnitude/t_steps

    force = []
    pres_nodes_idx = []
    m = 0

    for i in range(len(pres_nodes_full)):

        p_node_force_direction = force_dirs[i, :]

        for j in range(t_steps):

            updated_mag = step_mag * (j+1)

            force.append(updated_mag)
            force.append(p_node_force_direction[0])
            force.append(p_node_force_direction[1])
            force.append(p_node_force_direction[2])

            # ----------------------------------------

            idx = (pres_nodes_full[i] - 1) + (m * num_nodes)
            pres_nodes_idx.append(idx)
            m = m + 1

    force = np.reshape(force, (-1, 4)) # columns of this matrix are: magnitude, F_x, F_y, F_z

    # print(pres_nodes_idx)
    # print(force)


    force_length = t_steps * len(pres_nodes_full) * num_nodes


    M = np.zeros((force_length, 4))

    # print(pres_nodes_idx)

    for k in range(len(pres_nodes_idx)):

        # print('------------')
        # print(pres_nodes_idx[k])
        # print(force[k, 0])
        # print(force[k, 1])
        # print(force[k, 2])
        # print(force[k, 3])
        # print('------------')

        M[pres_nodes_idx[k], 0] = force[k, 0]
        M[pres_nodes_idx[k], 1] = force[k, 1]
        M[pres_nodes_idx[k], 2] = force[k, 2]
        M[pres_nodes_idx[k], 3] = force[k, 3]

    # print('Printing M')
    # print(M)

    df_f_2 = pd.DataFrame(data=M, columns=["Magnitude", "F_x", "F_y", "F_z"])

    # print(df_f_1)
    # print(df_f_2)


    frame_2 = [df_f_1, df_f_2]

    df_features = pd.concat(frame_2, axis=1)

    # print(df_features)

    return df_features

# --------------------------------------------------------------------------------------------------


def graph_indicator(num_nodes, num_p_nodes, num_dirs, t_steps):

    num_graphs = num_p_nodes * num_dirs * t_steps

    graph_labels = []

    for i in range(num_graphs):
        for j in range(num_nodes):
            graph_labels.append(i+1)

    df = pd.DataFrame(graph_labels)

    return df

# --------------------------------------------------------------------------------------------------

def graph_indicator_2(num_nodes, num_dirs, t_steps):

    num_graphs =  num_dirs * t_steps

    graph_labels = []

    for i in range(num_graphs):
        for j in range(num_nodes):
            graph_labels.append(i+1)

    df = pd.DataFrame(graph_labels)

    return df

# --------------------------------------------------------------------------------------------------


def graph_label(num_p_nodes, num_dirs, t_steps):

    num_graphs = num_p_nodes * num_dirs * t_steps

    graph_labels = []

    for i in range(num_graphs):
        graph_labels.append(i+1)

    df = pd.DataFrame(graph_labels)

    return df

# --------------------------------------------------------------------------------------------------

def graph_label_2(num_dirs, t_steps):

    num_graphs =  num_dirs * t_steps

    graph_labels = []

    for i in range(num_graphs):
        graph_labels.append(i+1)

    df = pd.DataFrame(graph_labels)

    return df

# --------------------------------------------------------------------------------------------------


def output_format(filename_output, t_steps): # this part needs to be hardcoded. This code is for when we have 5 time steps
    '''
    This function requires to be hardcoded.
    :param filename_output: an output file holding displacement values at x y z for t_steps
    :param t_steps: number of time steps in the FEA
    :return: a panda data frame
    '''


    output_v1 = genfromtxt(filename_output, delimiter=',')

    start = []
    end = []

    for i in range(t_steps):

        x = 3 * (i + 1)
        start.append(x)
        end.append(x + 3)

    out_1 = output_v1[:, start[0]: end[0]]
    out_2 = output_v1[:, start[1]: end[1]]
    out_3 = output_v1[:, start[2]: end[2]]
    out_4 = output_v1[:, start[3]: end[3]]
    out_5 = output_v1[:, start[4]: end[4]]
    out_6 = output_v1[:, start[5]: end[5]]
    out_7 = output_v1[:, start[6]: end[6]]
    out_8 = output_v1[:, start[7]: end[7]]
    out_9 = output_v1[:, start[8]: end[8]]
    out_10 = output_v1[:, start[9]: end[9]]
    out_11 = output_v1[:, start[10]: end[10]]
    out_12 = output_v1[:, start[11]: end[11]]
    out_13 = output_v1[:, start[12]: end[12]]
    out_14 = output_v1[:, start[13]: end[13]]
    out_15 = output_v1[:, start[14]: end[14]]
    out_16 = output_v1[:, start[15]: end[15]]
    out_17 = output_v1[:, start[16]: end[16]]
    out_18 = output_v1[:, start[17]: end[17]]
    out_19 = output_v1[:, start[18]: end[18]]
    out_20 = output_v1[:, start[19]: end[19]]
    out_21 = output_v1[:, start[20]: end[20]]
    out_22 = output_v1[:, start[21]: end[21]]
    out_23 = output_v1[:, start[22]: end[22]]
    out_24 = output_v1[:, start[23]: end[23]]
    out_25 = output_v1[:, start[24]: end[24]]
    out_26 = output_v1[:, start[25]: end[25]]
    out_27 = output_v1[:, start[26]: end[26]]
    out_28 = output_v1[:, start[27]: end[27]]
    out_29 = output_v1[:, start[28]: end[28]]
    out_30 = output_v1[:, start[29]: end[29]]

    df_1 = pd.DataFrame(data=out_1)
    df_2 = pd.DataFrame(data=out_2)
    df_3 = pd.DataFrame(data=out_3)
    df_4 = pd.DataFrame(data=out_4)
    df_5 = pd.DataFrame(data=out_5)
    df_6 = pd.DataFrame(data=out_6)
    df_7 = pd.DataFrame(data=out_7)
    df_8 = pd.DataFrame(data=out_8)
    df_9 = pd.DataFrame(data=out_9)
    df_10 = pd.DataFrame(data=out_10)
    df_11 = pd.DataFrame(data=out_11)
    df_12 = pd.DataFrame(data=out_12)
    df_13 = pd.DataFrame(data=out_13)
    df_14 = pd.DataFrame(data=out_14)
    df_15 = pd.DataFrame(data=out_15)
    df_16 = pd.DataFrame(data=out_16)
    df_17 = pd.DataFrame(data=out_17)
    df_18 = pd.DataFrame(data=out_18)
    df_19 = pd.DataFrame(data=out_19)
    df_20 = pd.DataFrame(data=out_20)
    df_21 = pd.DataFrame(data=out_21)
    df_22 = pd.DataFrame(data=out_22)
    df_23 = pd.DataFrame(data=out_23)
    df_24 = pd.DataFrame(data=out_24)
    df_25 = pd.DataFrame(data=out_25)
    df_26 = pd.DataFrame(data=out_26)
    df_27 = pd.DataFrame(data=out_27)
    df_28 = pd.DataFrame(data=out_28)
    df_29 = pd.DataFrame(data=out_29)
    df_30 = pd.DataFrame(data=out_30)

    frame = [df_1, df_2, df_3, df_4,
             df_5, df_6, df_7, df_8,
             df_9, df_10, df_11, df_12,
             df_13, df_14, df_15, df_16,
             df_17, df_18, df_19, df_20,
             df_21, df_22, df_23, df_24,
             df_25, df_26, df_27, df_28,
             df_29, df_30]

    df_output = pd.concat(frame)

    df_output.reset_index(drop=True, inplace=True)

    return df_output

