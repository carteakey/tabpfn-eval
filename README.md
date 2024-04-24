# mds-tmu-dl


Sliding window attention

https://github.com/automl/TabPFN/tree/main/tabpfn
[TabPFNDemo - Colaboratory (google.com)](https://colab.research.google.com/drive/194mCs6SEPEW6C0rcP7xWzcEtt1RBc8jJ)
[GitHub - automl/tabpfn-client](https://github.com/automl/tabpfn-client)
[HyperFast/hyperfast/hyperfast.py at main · AI-sandbox/HyperFast · GitHub](https://github.com/AI-sandbox/HyperFast/blob/main/hyperfast/hyperfast.py)


https://github.com/fmohr/tabpfnvsnaiveautoml/tree/main?tab=readme-ov-file
https://www.reddit.com/r/MachineLearning/comments/ya9r9l/d_tabpfn_a_transformer_that_solves_small_tabular/
https://twitter.com/FrankRHutter/status/1583410845307977733
https://www.automl.org/tabpfn-a-transformer-that-solves-small-tabular-classification-problems-in-a-second/


docker build -t latex . 


What is prior_bag (I guess it sequentially samples or something?)?
It mixes different prior models, i.e. it randomly selects either one model or another. In the repo, the Gaussian process prior is mixed with the SCM and BNN prior. However, in the config the weight of the GP prior is set to 0 however so the prior_bag does not have a function in that case.


Am I understanding correctly that the DifferentiableHyperparameter code is used only to sample dataset parameters from ranges set in the configs (and described in table 5 of the appendix)?

Yes exactly. In the configs dict in key differentiable_hyperparameters, there are the keys of hyperparameters that are sampled alongside distributions from which they are sampled. The sampled hyperparameters overwrite the values in the config dictionary (i.e. outside the differentiable_hyperparameters key) after they are sampled.


You are right the PriorFittingCustomPrior.ipynb notebook is used for training and the get_model function trains the model. The configs set in PriorFittingCustomPrior.ipynb reproduce our model. The configs are first retrieved from the function get_prior_config (scripts.model_configs) and then some settings are overwritten in the notebook.


Our trained models / classifiers save the parameters used for training and you can retrieve these parameters using:

classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=32, base_path=tabpfn_path)
classifier.c
Relevant for the used batch size is: batch_size (8), aggregate_k_gradients (8) and the number of parallel devices used (8, which is not saved). Multiplying these gives us the correct batch size 512.

https://github.com/automl/TabPFN/issues/26

I think, this might be a confusion that: i) we use 8 GPU training and the batch_size parameter is per GPU, ii) our aggregate_k_gradients aggregates seperate batches in our repo logic, which are the same batch in the optimizer logic, which we write about, though.


No of datasets: 247
Mean ROC: 0.843
Mean Cross Entropy: nan
Mean Accuracy: 0.802
Mean Prediction Time: 0.357s
 
No of datasets: 247
Mean ROC: 0.837
Mean Cross Entropy: 0.451
Mean Accuracy: 0.792
Mean Prediction Time: 0.39s

To identify hyperparameters that can be changed for better performance, it's important to have a clear understanding of the problem you are trying to solve and the specific model you are using. However, here are some hyperparameters in your code that you can consider modifying:

1. `config["epochs"]`: This hyperparameter determines the number of training epochs. Increasing the number of epochs may improve model performance, but be cautious of overfitting.

2. `config["max_num_classes"]`: This hyperparameter defines the maximum number of classes. Adjusting this value based on your dataset can improve model accuracy.

3. `config["sampling"]`: This hyperparameter determines the sampling strategy. Experimenting with different sampling techniques, such as "mixed" or "uniform", may lead to better results.

4. `config["categorical_feature_p"]`: This hyperparameter controls the probability of categorical features. Adjusting this value based on the nature of your data can improve model performance.

5. `config["emsize"]`: This hyperparameter sets the embedding size. Increasing the embedding size may capture more complex patterns in the data, but it may also increase model complexity.

6. `config["bptt"]`: This hyperparameter defines the number of time steps for backpropagation through time. Adjusting this value can affect the model's ability to capture long-term dependencies.

7. `config["batch_size"]`: This hyperparameter determines the number of samples processed in each training iteration. Increasing the batch size can speed up training, but it may also require more memory.

8. `config["epochs"]`: This hyperparameter sets the total number of training epochs. Adjusting this value can affect the convergence of the model.

Please note that the impact of these hyperparameters may vary depending on your specific problem and dataset. It's recommended to experiment with different values and evaluate the model's performance to find the optimal configuration.