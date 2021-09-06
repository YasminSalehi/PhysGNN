from library_imports import *


def get_data(pickled_file_path, dataset):
    with open(pickled_file_path, 'rb') as f:
        dataset_partial = torch.load(f)

        for i, data in enumerate(dataset_partial):
            feature = data.x
            edge_index = data.edge_index

            # print('---------------')
            # print(edge_index)
            # print(max(edge_index[0,:]))
            # print(max(edge_index[1,:]))
            # print('---------------')

            edge_values = data.edge_attr
            pos = data.pos
            y = data.y

            # create the data object
            data_obj = Data(x=feature, edge_index=edge_index, edge_attr=edge_values, pos=pos, y=y)
            dataset.append(data_obj)

    return dataset


def load_data(dataset_name):
    dataset = []

    print(dataset_name + ' is being loaded...')

    dataset = get_data(dataset_name + '_pickle/' + dataset_name + '_a_raw.pickle', dataset)
    print('1/11 loaded.')
    dataset = get_data(dataset_name + '_pickle/' + dataset_name + '_b_raw.pickle', dataset)
    print('2/11 loaded.')
    dataset = get_data(dataset_name + '_pickle/' + dataset_name + '_c_raw.pickle', dataset)
    print('3/11 loaded.')
    dataset = get_data(dataset_name + '_pickle/' + dataset_name + '_d_raw.pickle', dataset)
    print('4/11 loaded.')
    dataset = get_data(dataset_name + '_pickle/' + dataset_name + '_e_raw.pickle', dataset)
    print('5/11 loaded.')
    dataset = get_data(dataset_name + '_pickle/' + dataset_name + '_f_raw.pickle', dataset)
    print('6/11 loaded.')
    dataset = get_data(dataset_name + '_pickle/' + dataset_name + '_g_raw.pickle', dataset)
    print('7/11 loaded.')
    dataset = get_data(dataset_name + '_pickle/' + dataset_name + '_h_raw.pickle', dataset)
    print('8/11 loaded.')
    dataset = get_data(dataset_name + '_pickle/' + dataset_name + '_i_raw.pickle', dataset)
    print('9/11 loaded.')
    dataset = get_data(dataset_name + '_pickle/' + dataset_name + '_j_raw.pickle', dataset)
    print('10/11 loaded.')
    dataset = get_data(dataset_name + '_pickle/' + dataset_name + '_k_raw.pickle', dataset)
    print('11/11 loaded.')

    print(len(dataset))
    print('Datasets have been concatenated.')

    return dataset


def add_new_features(dataset):
    # Original Features:
    # Physiacal Property | Fx | Fy | Fz
    # New Features:
    # ------------------------------------------------------
    # Physical Property | Fx | Fy | Fz | ro | phi | theta
    # ro    = sqrt(Fx^2 + Fy^2 + Fz^2)
    # phi   = cos^-1(Fz / ro)
    # theta = sin^-1(Fy/ ro * sin(phi))
    # ------------------------------------------------------
    len_dataset = len(dataset)

    for i, data in enumerate(dataset):
        ro = torch.sqrt((data.x[:, 1] ** 2) + (data.x[:, 2] ** 2) + (data.x[:, 3] ** 2)).unsqueeze(-1)
        phi = torch.acos(torch.div(data.x[:, 3].unsqueeze(-1), ro))
        theta = torch.asin(torch.div(data.x[:, 2].unsqueeze(-1), torch.mul(ro, torch.sin(phi))))
        data.x = torch.cat((data.x, ro, phi, theta), dim=1)
        data.x = torch.nan_to_num(data.x)

        print(data.x.shape)
        print(str(i + 1) + '/' + str(len_dataset))

    print('New features have been added.')
    return dataset


def mean_and_std_calc(dataset):
    len_dataset = len(dataset)

    feature = torch.empty(1, 1)
    i = 0

    for i, data in enumerate(dataset):
        if i == 0:
            feature = data.x
        else:
            feature = torch.cat((feature, data.x), 0)
        i = i + 1

        print(str(i) + '/' + str(len_dataset))
        print(feature.shape)

    mean_val = torch.mean(feature, dim=0, keepdim=True)
    std_val = torch.std(feature, dim=0, keepdim=True)

    print(mean_val)
    print(std_val)

    return mean_val, std_val


def data_normalize(dataset):
    len_dataset = len(dataset)

    mean_val, std_val = mean_and_std_calc(dataset)

    for i, data in enumerate(dataset):
        data.x[:, 1:] = (data.x[:, 1:] - mean_val[:, 1:]) / std_val[:, 1:]
        # data.x[:,:] = (data.x[:,:]-mean_val[:,:])/std_val[:,:]
        print(str(i + 1) + '' + str(len_dataset))

    print('Data has been normalized.')
    return dataset

def data_preprocessing(dataset):

    dataset = add_new_features(dataset)
    dataset = data_normalize(dataset)

    return dataset