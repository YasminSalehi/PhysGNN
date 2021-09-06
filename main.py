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


# ----- Final Run ------

# config_selected = 'config1'     #
# config_selected = 'config2'     #
config_selected = 'config3'     #
# config_selected = 'config4'     #
# config_selected = 'config5'     #
# config_selected = 'config6'     #
# config_selected = 'config7'     #
# config_selected = 'config8'     #
# config_selected = 'config9'     #
# config_selected = 'config10'    #
# config_selected = 'config11'    #
# config_selected = 'config12'    #

model = train(dataset, writer, dataset_name, config_selected)
