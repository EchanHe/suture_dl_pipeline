# Suture pipeline

This pipeline aims to let people test different Deep Learning models on how to segment cranial sutures from micro CT scans of mammal skulls. Users can test different models and configurations (e.g. model hyper-parameters), and visualise the training and predicting process.

## Installation

Install anaconda python: https://www.anaconda.com/products/distribution

Open Anaconda Prompt. Windows: Search Anaconda Prompt.

Setup a virtual environment in conda and install all the required libraries (libraries are listed in the `env.yml file`).
`conda env create --file ./env.yml`

Activate the installed environment (default environment name is torch): `conda activate torch`


## Input images and input labels

Input images need to be all saved in one folder. The input labels (masks) need to be save in another folder, and a mask should be named same as to its corresponding image. Users should make sure that their datasets follow the data setting. A dataset need to be manipulated to follow the setting. A Jupyter notebook file `process_training_set.ipynb` can help with preparing users' datasets.

## Preprocessing images


The `process_training_set.ipynb` file allows users to batch process images and annotations. Currently, it supports renaming, copy files, padding. 

## Use it for training
Please change the `train_config.ini`, for example: 
- The directory of the images (`image_dir`. in the `[DIR]` section)
- The learning rate of the model (e.g. `learning_rate` in the `[PARAMS]` section)

Run `python train.py train_config.ini` in the anaconda prompt. You can also create your own configuration file and run `python train.py <config name>.ini` 

By default, the training log (e.g. training loss and validation loss) will be saved in the `./runs` folder. You can check the logs by running `tensorboard --logdir=runs` in the anaconda prompt. After running tensorboard successfully, use browse to open the address (the default should be http://localhost:6006/) shown in the prompt. You can then check and compare the different training runs. A log is named `<month and data>_<time>_<name of the computer>`.

Current models:
- U-net
- DeepLab
## Use it for predicting

Please change the `train_config.ini`, for example: 
- The directory of the the saved model (`checkpoint_path`. in the `[DIR]` section)

Run `python predict.py pred_config.ini` in the anaconda prompt. You can also create your own configuration file and run `python predict.py <config name>.ini`.

The output masks will be saved in the `output_path` of the configuration file. The mask's pixel values are: 0 - background, 1 - first class; 2- second class; n- nth class. Therefore, viewing a mask using the default image viewer of Windows will only see a black image. It's recommended to view the mask with tools such as ImageJ.

Output masks with better visualization will be save in the `output_vis_path`.


## Jupyter notebook version

To let the user have a more interactive session, users can use the `training.ipynb` and `predicting.ipynb`.