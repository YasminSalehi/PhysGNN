# PhysGNN

* The presented code in this repository is the implementation of PhysGNN.
* The code used for generating the datasets used in the paper is also included. 

* The tumour and brain model are taken from the paper: "A machine learning approach for real-time modelling of tissue deformation 
in image-guided neurosurgery" by Tonutti et al, available at: 
https://github.com/michetonu/MALTIDEM--Machine-Learning-for-Tissue-Deformation-Modelling

* The `Graph_load_batch` function in `pg_dataset.py` has been adapted from the `Graph_load_batch` function in 
`dataset.py` of Position-aware Graph Neural Networks code by You et al. available at:
https://github.com/JiaxuanYou/P-GNN

## Data Generation

1) Please download: 

    1) FEBio 2.9.1 @ https://febio.org/febio/febio-downloads/

    2) export_fig-master @ https://www.mathworks.com/matlabcentral/fileexchange/23629-export_fig

    3) GIBBON @ https://www.gibboncode.org/Installation/

    and place all the downloaded files in the "code" folder. 
    
2) Please make the following installations:
    1) `pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html`
    2) `pip install tensorboardX`
    3) `pip install networkx`

3) Please run `data_generator_1.m` to generate Dataset 1, and `data_generator_1.m` to generate Dataset 2.
Formatted data will be saved in dataset_1 and dataset_2 folders. 
4) Run `dataset_full.py` to generate the preprocessed data. 
5) Run `pg_dataset.py` to create 1/11 of dataset1/dataset2 and pickle them. Each run requires 22GB of RAM. 
Dataset 1 and Dataset 2 use up 16.79 GB of RAM each. 

## PhysGNN Training

1) Select the desired configuration from training in `main.py` and simply run the code. 

## Reproducing Results
1) Select the desired configuration from `reproduce.py` and simply run the code. 

## To run the code on Colab

(**Requirement**: A Google Colab Pro Account) 

### Training PhysGNN on Colab
1) Upload dataset_1 and dataset_2 folders to your Google Drive.
2) Upload `Dataset_Generation.ipynb` to your Google Drive. 
3) Set up the paths on your Colab notebook. 
4) Run all the cells consecutively. Ensure proper selection of the dataset to be configured as a pytorch 
geometric dataset and its corresponding pickle name. 
5) After one pytorch geometric dataset is created, restart the runtime to clear the RAM. 
6) Each section of Dataset 1 and Dataset 2 takes 1 hour and 15 minutes to be generated. 
7) After successful generation of the datasets, upload `PhysGNN.ipynb` to your Google Drive. 
8) Run all the cells in a consecutive order. 
    1) In the cell "Final Run for Training" select the dataset you wish to use
    2) In the "Final Run" section select the configuration you want to train.

### To reproduce the results on Colab 
1) Run all the cells until "Final Run for Training". Run the "Reproducing the Results" cell instead. 
2) Select the Dataset and the model you wish to reproduce their results in the "Final Run" under the Reproducibility 
cell. 
3) Results generated from the pickled configurations can be saved by setting the `save` parameter to 1. 
4) Setting `mean_mag_results` to 1 generates the mean Euclidean errors reported (Table 5). 
5) Setting `max_error_results` to 1 generates the max error results reported (Table 6).