from library_imports import *
from models import *
from main import *


# Early Stopping
def early_stopping(val_loss, current_min):
    stop = False
    if val_loss[-1] > current_min:
        stop = True
    return stop


###############################################################################################

# Training
def train(dataset, writer, dataset_name, config_selected):
    data_size = len(dataset)
    print(dataset_name)
    # ------------------------------------------------------------------------------------------------------------------
    if dataset_name == 'dataset_1':
        print('Dataset 1 is being used.')
        loader = DataLoader(dataset[:int(data_size * 0.7)], batch_size=4, shuffle=True)
        validation_loader = DataLoader(dataset[int(data_size * 0.7): int(data_size * 0.9)], batch_size=1, shuffle=False)
        test_loader = DataLoader(dataset[int(data_size * 0.9):], batch_size=1, shuffle=False)
    else:
        print('Dataset 2 is being used.')
        loader = DataLoader(dataset[:int(data_size * 0.7)], batch_size=8, shuffle=True)
        validation_loader = DataLoader(dataset[int(data_size * 0.7): int(data_size * 0.9)], batch_size=1, shuffle=False)
        test_loader = DataLoader(dataset[int(data_size * 0.9):], batch_size=1, shuffle=False)
    # ------------------------------------------------------------------------------------------------------------------
    # MODIFICATION #1 Build the Model
    if config_selected == 'config1':
        model = config1(7, 112, 3)
        print('Configuration 1 is selected.')
    elif config_selected == 'config2':
        model = config2(7, 112, 3)
        print('Configuration 2 is selected.')
    elif config_selected == 'config3':
        model = config3(7, 112, 3)
        print('Configuration 3 is selected.')
    elif config_selected == 'config4':
        model = config4(7, 112, 3)
        print('Configuration 4 is selected.')
    elif config_selected == 'config5':
        model = config5(7, 112, 3)
        print('Configuration 5 is selected.')
    elif config_selected == 'config6':
        model = config6(7, 112, 3)
        print('Configuration 6 is selected.')
    elif config_selected == 'config7':
        model = config7(7, 112, 3)
        print('Configuration 7 is selected.')
    elif config_selected == 'config8':
        model = config8(7, 112, 3)
        print('Configuration 8 is selected.')
    elif config_selected == 'config9':
        model = config9(7, 112, 3)
        print('Configuration 9 is selected.')
    elif config_selected == 'config10':
        model = config10(7, 112, 3)
        print('Configuration 10 is selected.')
    elif config_selected == 'config11':
        model = config11(7, 112, 3)
        print('Configuration 11 is selected.')
    elif config_selected == 'config12':
        model = config12(7, 112, 3)
        print('Configuration 12 is selected.')
    else:
        raise NotImplementedError
    # ------------------------------------------------------------------------------------------------------------------
    # Move model to GPU
    model = model.to(device)
    # ------------------------------------------------------------------------------------------------------------------
    opt = optim.AdamW(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5, min_lr=0.00000001,
                                                           verbose=True)
    # ------------------------------------------------------------------------------------------------------------------
    # print('parameters')
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print (name, param.data)
    # print('----------------')
    # ------------------------------------------------------------------------------------------------------------------
    # for plotting
    learning_curve_train = []
    learning_curve_val = []
    # ------------------------------------------------------------------------------------------------------------------
    # for early stopping
    patience = 0
    # ------------------------------------------------------------------------------------------------------------------
    # MODIFICATION #2 Chossing the path to save the trained model
    if dataset_name == 'dataset_1':
        file_path = "Results_1/"
    else:
        file_path = "Results_2/"
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    PATH = file_path + config_selected + '.pt'
    print(PATH)
    # ------------------------------------------------------------------------------------------------------------------
    learn_value_train = 0
    learn_value_validation = 0
    # train
    for epoch in range(15000):

        lr = scheduler.optimizer.param_groups[0]['lr']
        print(lr)

        total_loss = 0
        total_len = 0

        model.train()

        for batch in loader:
            batch = batch.to(device)

            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label)
            loss = loss / len(batch.y)
            loss.backward()
            opt.step()
            # total_loss += loss.item()  # is loss isn't being averaged
            total_loss += loss.item() * len(batch.y)  # if loss is being averaged
            total_len += len(label)

        total_loss /= total_len  # train loss
        # ------------------------------------------------------------------------------------------------------------------
        # Validation
        val_loss = val(validation_loader, model)
        learning_curve_train.append(total_loss)
        learning_curve_val.append(val_loss)
        print("Epoch {}. Train Loss: {:.16f}. Validation Loss: {:.16f}".format(epoch, total_loss, val_loss))
        # ------------------------------------------------------------------------------------------------------------------
        scheduler.step(val_loss)
        # ------------------------------------------------------------------------------------------------------------------
        # Saving model and early stopping
        if epoch != 0:
            stop = early_stopping(learning_curve_val, current_min)
            if stop == False:
                current_min = learning_curve_val[-1]
                torch.save(model.state_dict(), PATH.format(epoch))
                patience = 0
            else:
                patience = patience + 1
                print(patience)
                if patience == 15:
                    break
        else:
            current_min = learning_curve_val[0]
    # ------------------------------------------------------------------------------
    # MODIFICATION #3
    if config_selected == 'config1':
        final_model = config1(7, 112, 3)
        print('Configuration 1 best state loading...')
    elif config_selected == 'config2':
        final_model = config2(7, 112, 3)
        print('Configuration 2 best state loading...')
    elif config_selected == 'config3':
        final_model = config3(7, 112, 3)
        print('Configuration 3 best state loading...')
    elif config_selected == 'config4':
        final_model = config4(7, 112, 3)
        print('Configuration 4 best state loading...')
    elif config_selected == 'config5':
        final_model = config5(7, 112, 3)
        print('Configuration 5 best state loading...')
    elif config_selected == 'config6':
        final_model = config6(7, 112, 3)
        print('Configuration 6 best state loading...')
    elif config_selected == 'config7':
        final_model = config7(7, 112, 3)
        print('Configuration 7 best state loading...')
    elif config_selected == 'config8':
        final_model = config8(7, 112, 3)
        print('Configuration 8 best state loading...')
    elif config_selected == 'config9':
        final_model = config9(7, 112, 3)
        print('Configuration 9 best state loading...')
    elif config_selected == 'config10':
        final_model = config10(7, 112, 3)
        print('Configuration 10 best state loading...')
    elif config_selected == 'config11':
        final_model = config11(7, 112, 3)
        print('Configuration 11 best state loading...')
    elif config_selected == 'config12':
        final_model = config12(7, 112, 3)
        print('Configuration 12 best state loading...')
    else:
        raise ValueError('The selected configuration does not exist.')
    # --------------------------------------------------------------------------
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    final_model.load_state_dict(torch.load(PATH))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Final model to GPU
    final_model = final_model.to(device)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    lowest_val_loss = val(validation_loader, final_model)
    test_loss = test(test_loader, final_model)

    print("Validation Loss: {:.16f}. Test Loss: {:.16f}".format(lowest_val_loss, test_loss))
    print('---------------')
    # ------------------------------------------------------------------------------------------------------------------
    for j in range(len(learning_curve_train)):
        learn_value_train = learning_curve_train[j]
        learn_value_validation = learning_curve_val[j]
        print("{:.16f} {:.16f}".format(learn_value_train, learn_value_validation))

    return final_model


###############################################################################################
# Validation

def val(loader, model):
    model.eval()

    error = 0
    total_loss = 0
    total_length = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            pred = model(batch)
            label = batch.y
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # error = F.mse_loss(pred, label) # MSE
            # total_loss += error.item() * len(label) #MSE
            # ------------------------------------------------------------------
            # total_loss += (pred - label).abs().sum().item()  #MAE
            # ------------------------------------------------------------------
            total_loss += (torch.sqrt(
                ((pred[:, 0] - label[:, 0]) ** 2).unsqueeze(-1) + ((pred[:, 1] - label[:, 1]) ** 2).unsqueeze(-1) + (
                            (pred[:, 2] - label[:, 2]) ** 2).unsqueeze(-1))).sum().item()  # MAE for magnitude
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            total_length += len(label)

    total_loss /= total_length

    return total_loss


###############################################################################################
# Testing

def test(loader, model):
    model.eval()

    error = 0
    total_loss = 0
    total_length = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            pred = model(data)
            label = data.y
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # error = F.mse_loss(pred, label) # MSE
            # total_loss += error.item() * len(label) #MSE
            # ------------------------------------------------------------------
            # total_loss += (pred - label).abs().sum().item()  #MAE
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            total_loss += (torch.sqrt(
                ((pred[:, 0] - label[:, 0]) ** 2).unsqueeze(-1) + ((pred[:, 1] - label[:, 1]) ** 2).unsqueeze(-1) + (
                            (pred[:, 2] - label[:, 2]) ** 2).unsqueeze(-1))).sum().item()  # MAE for magnitude
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            total_length += len(label)

    total_loss /= total_length

    return total_loss


###############################################################################################


def val_reproduce(loader, model):
    model.eval()
    error = 0
    total_loss = 0
    total_length = 0

    actual = torch.empty(1, 1)
    prediction = torch.empty(1, 1)
    i = 0

    with torch.no_grad():
        for batch in loader:

            batch = batch.to(device)

            pred = model(batch)
            label = batch.y
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # to save prediction and actual labels
            if i == 0:
                prediction = pred
                actual = label
                i = i + 1
            else:
                prediction = torch.cat((prediction, pred), 0)
                actual = torch.cat((actual, label), 0)
                i = i + 1
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # error = F.mse_loss(pred, label) # MSE
            # total_loss += error.item() * len(label) #MSE
            # ------------------------------------------------------------------
            # total_loss += (pred - label).abs().sum().item()  #MAE
            # ------------------------------------------------------------------
            total_loss += (torch.sqrt(
                ((pred[:, 0] - label[:, 0]) ** 2).unsqueeze(-1) + ((pred[:, 1] - label[:, 1]) ** 2).unsqueeze(-1) + (
                            (pred[:, 2] - label[:, 2]) ** 2).unsqueeze(-1))).sum().item()  # MAE for magnitude
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            total_length += len(label)

    total_loss /= total_length

    return total_loss, prediction, actual


###############################################################################################


def test_reproduce(loader, model):
    model.eval()
    error = 0
    total_loss = 0
    total_length = 0

    actual = torch.empty(1, 1)
    prediction = torch.empty(1, 1)
    i = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            label = data.y
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # to save prediction and actual labels
            if i == 0:
                prediction = pred
                actual = label
                i = i + 1
            else:
                prediction = torch.cat((prediction, pred), 0)
                actual = torch.cat((actual, label), 0)
                i = i + 1
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # error = F.mse_loss(pred, label) # MSE
            # total_loss += error.item() * len(label) #MSE
            # ------------------------------------------------------------------
            # total_loss += (pred - label).abs().sum().item()  #MAE
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            total_loss += (torch.sqrt(
                ((pred[:, 0] - label[:, 0]) ** 2).unsqueeze(-1) + ((pred[:, 1] - label[:, 1]) ** 2).unsqueeze(-1) + (
                            (pred[:, 2] - label[:, 2]) ** 2).unsqueeze(-1))).sum().item()  # MAE for magnitude
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            total_length += len(label)

    total_loss /= total_length

    return total_loss, prediction, actual


###############################################################################################


def max_magnitude_error(loader, model):
    model.eval()
    error = 0
    max_errors = []
    running_time = []
    max_displacement = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            t0 = time.time()
            pred = model(data)
            label = data.y
            t1 = time.time()
            total = t1 - t0

            running_time.append(total)

            loss = (torch.sqrt(
                ((pred[:, 0] - label[:, 0]) ** 2).unsqueeze(-1) + ((pred[:, 1] - label[:, 1]) ** 2).unsqueeze(-1) + (
                            (pred[:, 2] - label[:, 2]) ** 2).unsqueeze(-1)))  # MAE for magnitude

            max_loss = max(loss).item()
            print(max_loss)
            max_errors.append(max_loss)

            displacement = torch.sqrt(
                (label[:, 0] ** 2).unsqueeze(-1) + (label[:, 1] ** 2).unsqueeze(-1) + (label[:, 2] ** 2).unsqueeze(-1))
            max_displacement.append(max(displacement).item())

    return max_displacement, max_errors, running_time


def reproduce(dataset, writer, config_selected, save, mean_mag_results, max_error_results):
    data_size = len(dataset)
    # -----------------------------------------------------------------------------------------------
    if mean_mag_results == 1 and max_error_results == 0:
        validation_loader = DataLoader(dataset[int(data_size * 0.7): int(data_size * 0.9)], batch_size=8, shuffle=False)
        test_loader = DataLoader(dataset[int(data_size * 0.9):], batch_size=8, shuffle=False)
    else:
        validation_loader = DataLoader(dataset[int(data_size * 0.7): int(data_size * 0.9)], batch_size=1, shuffle=False)
        test_loader = DataLoader(dataset[int(data_size * 0.9):], batch_size=1, shuffle=False)
    # ------------------------------------------------------------------------------------------------------------------
    # Select the path to the file which holds the final model
    if dataset_name == 'dataset_1':
        file_path = "Results_1/"
    else:
        file_path = "Results_2/"
    # --------------------------------------------
    PATH = file_path + config_selected + '.pt'
    print(PATH)
    # --------------------------------------------
    # Initialize the model
    if config_selected == 'config1':
        final_model = config1(7, 112, 3)
        print('Configuration 1 has been selected.')
    elif config_selected == 'config2':
        final_model = config2(7, 112, 3)
        print('Configuration 2 has been selected.')
    elif config_selected == 'config3':
        final_model = config3(7, 112, 3)
        print('Configuration 3 has been selected.')
    elif config_selected == 'config4':
        final_model = config4(7, 112, 3)
        print('Configuration 4 has been selected.')
    elif config_selected == 'config5':
        final_model = config5(7, 112, 3)
        print('Configuration 5 has been selected.')
    elif config_selected == 'config6':
        final_model = config6(7, 112, 3)
        print('Configuration 6 has been selected.')
    elif config_selected == 'config7':
        final_model = config7(7, 112, 3)
        print('Configuration 7 has been selected.')
    elif config_selected == 'config8':
        final_model = config8(7, 112, 3)
        print('Configuration 8 has been selected.')
    elif config_selected == 'config9':
        final_model = config9(7, 112, 3)
        print('Configuration 9 has been selected.')
    elif config_selected == 'config10':
        final_model = config10(7, 112, 3)
        print('Configuration 10 has been selected.')
    elif config_selected == 'config11':
        final_model = config11(7, 112, 3)
        print('Configuration 11 has been selected.')
    elif config_selected == 'config12':
        final_model = config12(7, 112, 3)
        print('Configuration 12 has been selected.')
    else:
        raise NotImplementedError
        # ----------------------------------------------
    # Load the model
    final_model.load_state_dict(torch.load(PATH))
    # ----------------------------------------------
    # Final model to GPU
    final_model = final_model.to(device)
    # ----------------------------------------------
    # ######################################################################################################
    # Reproducing the mean of maximum errors in magnitude and runtime results
    if max_error_results == 1:
        max_displacement, max_error_test, running_time = max_magnitude_error(test_loader, final_model)

        max_maximum_displacement = max(max_displacement)
        print(max_maximum_displacement)
        mean_maximum_displacement = statistics.mean(max_displacement)
        print(mean_maximum_displacement)
        std_max_displacement = statistics.stdev(max_displacement)
        print(std_max_displacement)
        print("Maximum displacement in the test set: {:.4f}. Average maximum displacement {:.4f} +/- {:.4f}".format(
            max_maximum_displacement, mean_maximum_displacement, std_max_displacement))

        mean_maximum_test_loss = statistics.mean(max_error_test)
        std_maximum_test_loss = statistics.stdev(max_error_test)
        print(
            "Average maximum magnitude error: {:.4f} +/- {:.4f}".format(mean_maximum_test_loss, std_maximum_test_loss))

        mean_time = statistics.mean(running_time)
        std_time = statistics.stdev(running_time)
        print("Average running time: {:.4f} +/- {:.4f}".format(mean_time, std_time))

    # ######################################################################################################
    # Reproducing the results in Table 1 and Table 2
    if mean_mag_results == 1:

        if save == 0:
            lowest_val_loss = val(validation_loader, final_model)
            test_mse = test(test_loader, final_model)
            print("Validation Loss: {:.16f} Test Loss: {:.16f}".format(lowest_val_loss, test_mse))
        else:
            lowest_val_loss, prediction_val, actual_val = val_reproduce(validation_loader, final_model)
            test_mse, prediction_test, actual_test = test_reproduce(test_loader, final_model)
            print("Validation Loss: {:.16f} Test Loss: {:.16f}".format(lowest_val_loss, test_mse))

            # -----------------------------------------------------------------------------------------------
            # Saving the results for further analysis
            print('Saving the results...')

            # Validation set
            prediction_val = prediction_val.cpu()
            actual_val = actual_val.cpu()
            prediction_val = prediction_val.numpy()
            prediction_val_df = pd.DataFrame(prediction_val)
            # ------------------------------------------------------------------------------------------------
            prediction_val_df.to_csv(file_path + 'csv/val/prediction_' + config_selected + '.csv')
            # ------------------------------------------------------------------------------------------------
            actual_val = actual_val.numpy()
            actual_val_df = pd.DataFrame(actual_val)
            # ------------------------------------------------------------------------------------------------
            actual_val_df.to_csv(file_path + 'csv/val/actual_' + config_selected + '.csv')
            # ------------------------------------------------------------------------------------------------

            # Test set
            prediction_test = prediction_test.cpu()
            actual_test = actual_test.cpu()
            prediction_test = prediction_test.numpy()
            prediction_test_df = pd.DataFrame(prediction_test)
            # ------------------------------------------------------------------------------------------------
            prediction_test_df.to_csv(file_path + 'csv/test/prediction_' + config_selected + '.csv')
            # ------------------------------------------------------------------------------------------------
            actual_test = actual_test.numpy()
            actual_test_df = pd.DataFrame(actual_test)
            # ------------------------------------------------------------------------------------------------
            actual_test_df.to_csv(file_path + 'csv/test/actual_' + config_selected + '.csv')
            # ------------------------------------------------------------------------------------------------
