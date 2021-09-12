from preprocessing import *

# Data set 9: Applying forces to 100 nodes at a time. Forces have 30 different magnitudes and 15 different directions.

def dataset_2(idx):

    input_path = 'Data_Generator/code/input_2/'
    output_path = 'Data_Generator/code/output_2/csv/'

    # Initialization
    elements_filename = input_path + 'elements.csv'
    element_id_filename = input_path + 'element_ID.csv'
    xyz_filename = input_path + 'xyz.csv'
    support_nodes_filename = input_path + 'bcSupportList.csv'


    pres_nodes_filename = input_path + 'bcPrescribeList.csv'
    boundary_faces_filename = input_path + 'boundary_faces.csv'
    boundary_id_filename = input_path + 'boundary_marker.csv'

    num_nodes = 9118  # number of nodes per each graph
    num_p_nodes = 100
    num_dirs = 15
    t_steps = 30
    f_magnitude = 20  # units: Newtons

    if idx == 1:
        formatted_data_path = 'dataset_2/a/'
        filename_force_dir_x = input_path + 'force_dir_x_a.csv'
        filename_force_dir_y = input_path + 'force_dir_y_a.csv'
        filename_force_dir_z = input_path + 'force_dir_z_a.csv'
        offset_output = 0
    elif idx == 2:
        formatted_data_path = 'dataset_2/b/'
        filename_force_dir_x = input_path + 'force_dir_x_b.csv'
        filename_force_dir_y = input_path + 'force_dir_y_b.csv'
        filename_force_dir_z = input_path + 'force_dir_z_b.csv'
        offset_output = 15
    elif idx == 3:
        formatted_data_path = 'dataset_2/c/'
        filename_force_dir_x = input_path + 'force_dir_x_c.csv'
        filename_force_dir_y = input_path + 'force_dir_y_c.csv'
        filename_force_dir_z = input_path + 'force_dir_z_c.csv'
        offset_output = 30
    elif idx == 4:
        formatted_data_path = 'dataset_2/d/'
        filename_force_dir_x = input_path + 'force_dir_x_d.csv'
        filename_force_dir_y = input_path + 'force_dir_y_d.csv'
        filename_force_dir_z = input_path + 'force_dir_z_d.csv'
        offset_output = 45
    elif idx == 5:
        formatted_data_path = 'dataset_2/e/'
        filename_force_dir_x = input_path + 'force_dir_x_e.csv'
        filename_force_dir_y = input_path + 'force_dir_y_e.csv'
        filename_force_dir_z = input_path + 'force_dir_z_e.csv'
        offset_output = 60
    elif idx == 6:
        formatted_data_path = 'dataset_2/f/'
        filename_force_dir_x = input_path + 'force_dir_x_f.csv'
        filename_force_dir_y = input_path + 'force_dir_y_f.csv'
        filename_force_dir_z = input_path + 'force_dir_z_f.csv'
        offset_output = 75
    elif idx == 7:
        formatted_data_path = 'dataset_2/g/'
        filename_force_dir_x = input_path + 'force_dir_x_g.csv'
        filename_force_dir_y = input_path + 'force_dir_y_g.csv'
        filename_force_dir_z = input_path + 'force_dir_z_g.csv'
        offset_output = 90
    elif idx == 8:
        formatted_data_path = 'dataset_2/h/'
        filename_force_dir_x = input_path + 'force_dir_x_h.csv'
        filename_force_dir_y = input_path + 'force_dir_y_h.csv'
        filename_force_dir_z = input_path + 'force_dir_z_h.csv'
        offset_output = 105
    elif idx == 9:
        formatted_data_path = 'dataset_2/i/'
        filename_force_dir_x = input_path + 'force_dir_x_i.csv'
        filename_force_dir_y = input_path + 'force_dir_y_i.csv'
        filename_force_dir_z = input_path + 'force_dir_z_i.csv'
        offset_output = 120
    elif idx == 10:
        formatted_data_path = 'dataset_2/j/'
        filename_force_dir_x = input_path + 'force_dir_x_j.csv'
        filename_force_dir_y = input_path + 'force_dir_y_j.csv'
        filename_force_dir_z = input_path + 'force_dir_z_j.csv'
        offset_output = 135
    elif idx == 11:
        formatted_data_path = 'dataset_2/h/'
        filename_force_dir_x = input_path + 'force_dir_x_k.csv'
        filename_force_dir_y = input_path + 'force_dir_y_k.csv'
        filename_force_dir_z = input_path + 'force_dir_z_k.csv'
        offset_output = 150
    else:
        raise ValueError('Dataset for offsets above 11 do not exist.')

    ######################################################################################################
    # Output file names: 1) formatted input files, 2) intermediate files, 3) formatted output files
    # ** We only need to upload the formatted input and output files to Google Drive

    # 1) formatted input files
    adj_matrix_filename = formatted_data_path + 'A.csv'
    node_att_filename = formatted_data_path + 'node_attributes.csv'
    graph_indicator_filename = formatted_data_path + 'graph_indicator.csv'
    graph_labels_filename = formatted_data_path + 'graph_labels.csv'
    # -----------------------------------------------------------------------------
    # 2) intermediate files
    node_material_id_filename = formatted_data_path + 'node_material_id.csv'
    rigid_nodes_filename = formatted_data_path + 'rigid_nodes_id.csv'
    adj_matrix_short_filename = formatted_data_path + 'A_partial.csv'
    # -----------------------------------------------------------------------------
    # 3) formatted output files
    output_displacement_filename = formatted_data_path + 'output_displacement.csv'

    ###################################################################################################################

    # Creating the adjacency matrix

    A = adj_matrix_builder(elements_filename, num_nodes)
    adj_matrix_full_format(A, num_nodes, num_dirs, t_steps, adj_matrix_filename)

    ###################################################################################################################

    # Creating the feature matrix

    df_xyz = xyz(xyz_filename)
    df_mat_id = node_material_assign(elements_filename, element_id_filename, boundary_faces_filename, boundary_id_filename,
                                     num_nodes, node_material_id_filename)
    df_rigid_id = node_support_assign(support_nodes_filename, num_nodes, rigid_nodes_filename)

    df_features = feature_constructor_2(num_nodes, df_xyz, df_mat_id, df_rigid_id, pres_nodes_filename,
                              filename_force_dir_x, filename_force_dir_y, filename_force_dir_z, f_magnitude, t_steps)

    df_features.to_csv(node_att_filename, encoding='utf-8', header=False, index=False)

    ###################################################################################################################

    # Creating the Graph indicator file (which determines which node belongs to which graph)

    node_graph_labels = graph_indicator_2(num_nodes, num_dirs, t_steps)
    node_graph_labels.to_csv(graph_indicator_filename, header=False, index=False)

    # ###################################################################################################################
    #
    # # Graph labels

    graph_labels = graph_label_2(num_dirs, t_steps)
    graph_labels.to_csv(graph_labels_filename, header=False, index=False)

    # ###################################################################################################################
    #
    # Output for 100 load nodes, 15 directions, and 30 time steps. Force is applied to all nodes at once.

    output_files = []

    for i in range(1, num_dirs + 1):
        s1 = "displacement_nodes_dir_"
        s2 = str(i + offset_output)
        s3 = ".csv"
        s = output_path + s1 + s2 + s3
        print(s)
        output_files.append(s)

    df_1 = output_format(output_files[0], t_steps)
    df_2 = output_format(output_files[1], t_steps)
    df_3 = output_format(output_files[2], t_steps)
    df_4 = output_format(output_files[3], t_steps)
    df_5 = output_format(output_files[4], t_steps)
    df_6 = output_format(output_files[5], t_steps)
    df_7 = output_format(output_files[6], t_steps)
    df_8 = output_format(output_files[7], t_steps)
    df_9 = output_format(output_files[8], t_steps)
    df_10 = output_format(output_files[9], t_steps)
    df_11 = output_format(output_files[10], t_steps)
    df_12 = output_format(output_files[11], t_steps)
    df_13 = output_format(output_files[12], t_steps)
    df_14 = output_format(output_files[13], t_steps)
    df_15 = output_format(output_files[14], t_steps)

    frame = [df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9, df_10, df_11, df_12, df_13, df_14, df_15]

    df_output = pd.concat(frame)

    df_output.reset_index(drop=True, inplace=True)

    df_output.to_csv(output_displacement_filename, header=False, index=False)

    print(df_output)

    # ##################################################################################################################
    # ##################################################################################################################
    # ##################################################################################################################
    # ##################################################################################################################

    # FEATURE PREPROCESSING

    data_node_att = np.loadtxt(formatted_data_path + 'node_attributes.csv', delimiter=',')
    # --------------------------------------------------------------------------------------------------------------------
    # Node attribute format
    # {x} {y} {z} {material_ID} {Rigid} {Force magnitude} {F_x} {F_y} {F_z}
    #  0   1   2       3           4            5           6     7     8
    # material id 1: brain, 2: tumour
    # ----------------------------------------------------------------------------------------------------------------------

    # ############# Continuous encoding of rigid ID and material ID ###############
    physics_prop = np.zeros((data_node_att.shape[0], 1), dtype=float)

    for j in range(data_node_att.shape[0]):
        if data_node_att[j, 4] == 1:
            physics_prop[j, 0] = 0
        else:
            if data_node_att[j, 3] == 1:
                physics_prop[j, 0] = 1
            else:
                physics_prop[j, 0] = 0.4
    # ############################################################################
    # ############# Multiplication of force magnitude by direction ###############
    force_magnitude = data_node_att[:, 5]
    x_mag_and_direction = np.multiply(data_node_att[:, 6], force_magnitude)
    y_mag_and_direction = np.multiply(data_node_att[:, 7], force_magnitude)
    z_mag_and_direction = np.multiply(data_node_att[:, 8], force_magnitude)
    # ----------------------------------------------------------------------------------------------------------------------
    x_mag_and_direction = np.reshape(x_mag_and_direction, (-1, 1))
    y_mag_and_direction = np.reshape(y_mag_and_direction, (-1, 1))
    z_mag_and_direction = np.reshape(z_mag_and_direction, (-1, 1))

    # ----------------------------------------------------------------------------------------------------------------------
    feature_normalized = np.concatenate((data_node_att[:, 0:3], physics_prop, x_mag_and_direction, y_mag_and_direction,
                                         z_mag_and_direction), axis=1)
    # ----------------------------------------------------------------------------------------------------------------------
    print(feature_normalized)

    print(feature_normalized.shape)

    r = np.ptp(feature_normalized, axis=0)

    print(r)

    np.savetxt(formatted_data_path + "/node_attributes_raw.csv", feature_normalized, delimiter=",",
               fmt=('%f, %f, %f, %f, %f, %f, %f'))
