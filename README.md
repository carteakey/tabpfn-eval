# TabPFN Replicated

The [TabPFN](https://github.com/automl/TabPFN/) is a neural network that learned to do tabular data prediction. 

## Installation
Need Python 3.9 or below
```bash
conda create --name tabpfn python=3.9
```

Install all the libraries
```bash
pip install tabpfn[full]
# install additional requirements
pip install requirements.txt
```

To run the autogluon and autosklearn baseline, 
create a separate environment and install 

```bash
conda create --name tabpfn-automl python=3.9
```

```bash
pip install autosklearn==0.14.5
pip install autogluon==0.4.0
```

## Getting Started
See `src/tabpfn_demo.py` for scikit-learn interface.


```shell
src
├── __init__.py                         # initializing package
├── datasets.py                         # selecting the datasets from OpenML
├── eval_automl_models.py               # evaluating AutoML models
├── eval_baseline_models.py             # evaluating baseline models
├── eval_baseline_tabpfn.py             # evaluating baseline TabPFN model
├── eval_gbdt_models.py                 # evaluating GBDT models
├── eval_modified_tabpfn.py              # evaluating modified TabPFN model
├── data
│   ├── openml_baseline_tabpfn.csv      # baseline results for TabPFN
│   └── openml_modified_tabpfn.csv       # modified results for TabPFN
│   ├── openml_list.csv                 # list of selected datasets and attributes
├── models
│   └── tabpfn
│       ├── modified                      # modified / retrained TabPFN
│       │   ├── model_configs.py          ## model configurations  
│       │   ├── models_diff              ## model checkpoints 
│       │   ├── tabpfn_new_params.json   ## training parameters 
│       │   ├── train.py                 ## training script
│       │   ├── train_config.py           ## training configurations 
│       │   └── train_continue.py        ## continuing training from checkpoint

│       └── original                     # original TabPFN
│           ├── model_configs.py          ## model configurations  
│           ├── models_diff              ## model checkpoints 
│           ├── tabpfn_orig_params.json  ## training parameters 
│           ├── tabpfn_orig_params.py    ## extract original training params
│           ├── train.py                 ## training script
│           └── train_config.py           ## training configurations 
├── plots
│   ├── classifier_decision_boundary.py  # plot classifier decision boundary
│   ├── transformer_model_viz.py        # plot transformer model architecture
│   └── visualize_priors.py             # plot Priors
├── tabpfn_demo.py                      # TabPFN demo using scikit-learn interface
└── utils.py                            # utility functions
```
#### Running the code 

Please change BASE_DIR in `src/__init__.py`

Download the retrained/modified model - `prior_diff_real_checkpointkc_n_0_epoch_100.cpkt` and save it to `src/models/modified/models_diff`

###### Generating datasets
```
(tabpfn) kchauhan@kpc:~/repos/mds-tmu-dl$ python3 -m src.evals.datasets
```




## Evaluation

### TabPFN (original)
- No of datasets: 247
- Mean ROC: 0.843
- Mean Cross Entropy: nan
- Mean Accuracy: 0.802
- Mean Prediction Time: 0.357s
 
### TabPFN (retrained/modified) - trained for just 5 hours.
- No of datasets: 247
- Mean ROC: 0.837
- Mean Cross Entropy: 0.451
- Mean Accuracy: 0.792
- Mean Prediction Time: 0.39s

See `training_log.txt`

What is prior_bag (I guess it sequentially samples or something?)?
It mixes different prior models, i.e. it randomly selects either one model or another. In the repo, the Gaussian process prior is mixed with the SCM and BNN prior. However, in the config the weight of the GP prior is set to 0 however so the prior_bag does not have a function in that case.


Am I understanding correctly that the DifferentiableHyperparameter code is used only to sample dataset parameters from ranges set in the configs (and described in table 5 of the appendix)?

Yes exactly. In the configs dict in key differentiable_hyperparameters, there are the keys of hyperparameters that are sampled alongside distributions from which they are sampled. The sampled hyperparameters overwrite the values in the config dictionary (i.e. outside the differentiable_hyperparameters key) after they are sampled.


You are right the PriorFittingCustomPrior.ipynb notebook is used for training and the get_model function trains the model. The configs set in PriorFittingCustomPrior.ipynb reproduce our model. The configs are first retrieved from the function get_prior_config (scripts.model_configs) and then some settings are overwritten in the notebook.
