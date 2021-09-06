from library_imports import *
from utils import *
from train import *

random.seed(1)
torch.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('CUDA availability:', torch.cuda.is_available())
writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
# ------------------------------------------------------------------------------
# Specifying the dataset
# dataset_name = 'dataset_1'
dataset_name = 'dataset_2'
# ----------------------------------------------------
# Load data
dataset = load_data(dataset_name)
# ----------------------------------------------------
# Preprocess data
dataset = data_preprocessing(dataset)
# # ----------------------------------------------------
random.Random(1).shuffle(dataset)
print('Pytorch Geometric dataset has been shuffeled.')

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# ------Final Run -----------

# If save is equal to 1, the prediction and actual values will be saved for
# further result processing.
save = 0
mean_mag_results = 0
max_error_results = 1
# ----------------------------------------------------------------------------------------------------
# Please note:
# Best performing model for dataset 1 is config 9
# Best performing model for dataset 2 is config 3
# ----------------------------------------------------------------------------------------------------
# Select configuration

# reproduce(dataset, writer, 'config1', save, mean_mag_results, max_error_results)
# reproduce(dataset, writer, 'config2', save, mean_mag_results, max_error_results)
# reproduce(dataset, writer, 'config3', save, mean_mag_results, max_error_results) # best performance dataset 2
# reproduce(dataset, writer, 'config4', save, mean_mag_results, max_error_results)
# reproduce(dataset, writer, 'config5', save, mean_mag_results, max_error_results)
# reproduce(dataset, writer, 'config6', save, mean_mag_results, max_error_results)
# reproduce(dataset, writer, 'config7', save, mean_mag_results, max_error_results)
# reproduce(dataset, writer, 'config8', save, mean_mag_results, max_error_results)
reproduce(dataset, writer, 'config9', save, mean_mag_results, max_error_results) # best performance dataset 1
# reproduce(dataset, writer, 'config10', save, mean_mag_results, max_error_results)
# reproduce(dataset, writer, 'config11', save, mean_mag_results, max_error_results)
# reproduce(dataset, writer, 'config12', save, mean_mag_results, max_error_results)


