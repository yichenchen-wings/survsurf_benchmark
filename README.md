# SurvSurf_benchmark

## Dependnecies
Dependencies are listed in `env_spec_python/environment.yml` and `env_spec_python/pip_requirements.txt`. To encapsulate the dependencies, you should create a new `conda` or `mamba` environment and install the dependencies from `environment.yml` first. Then use the `pip` within that environment (you can check by running `which pip` to check if it is within the correct `conda`/`mamba` environment) to `pip install` the dependencies from `pip_requirements.txt`, which contains the SurvSurf model as an installable package. 

(Before the work has been published) You will be prompted to enter a 'password' (i.e. a git access token) in order to install the SurvSurf model, please contact the author or raise a git issue for access token.

## The (to be) published model
The (to be published model) is wrapped in the `SurvSurf2DTaddTGNormTG` class in `model_factory_survsurf.py`. This class can be modified to include additional encoder modules to transform `xs` (e.g. a CNN that converts image-based input into an embedding vector) before the parent `forward` function (i.e. `super().forward`) is called.

## Input data structure
Please refer to the example datasets included and the data-wrangling notebooks.

## To reproduce the study results from scratch
1. Run the data wrangling notebooks (data for the clinical trial NCT00981058 needs to be downloaded manually because it needs permission from Data Sphere)
2. Run train-test split notebooks
3. Edit the `train_script_<DeepHit or SurvSurf>_local.sh` scripts with path to the correct config file and run.
4. Run the `notebook_fit_sksurv_<blah>.ipynb` notebooks for the non-NN models.
5. Run the `notebook_Check_val_set_<blah>_ipcw_certain.ipynb` notebooks for test-set performance evaluation metrics. Can be toggled to run on the validations set if needed.
6. Run the `notebook_present_model_eval_metrics_by_datasets.ipynb` notebooks to generate the figures.

If the model is already trained, and data already prepared (will be for all datasets except the clinical trial), you only need to run step 4, 5, and 6 (will need to command out the clinical trial data sections).

The main python scripts that execute the training pipeline are contained in the `train_<SurvSurf or DeepHit>.py` files, which is called through the `train_script_<DeepHit or SurvSurf>_local.sh` scripts and use the config `.json` files to configure the training. The `pl_training.py` defines the behaviour of the trainer in `train_<SurvSurf or DeepHit>.py`.

The `model_factory_<deephit or survsurf>.py` files contains wrapper classes for the models that are compatible to `pytorch-ligtning`. Loss functions (the published ones and other exploratory ones) adapted to work with the specific model outputs are contained in these files as well. The wrapper classes are inherited from the class defined in `pl_wrapper.py`. 

The `dataset_<name>.py` files contain dataset classes that read the train-test split dataframes and transform them into the required format for training. There are several modes of transformations that are selectable for training and validation/testing. The config `.json` files and the model evaluation notebooks indicate which mode was used for training and validation/testing.



