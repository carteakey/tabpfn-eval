import time
from tabpfn.scripts.model_builder import get_model, save_model
import torch 
import os 

device = "cuda"
base_path = "."
max_features = 100
maximum_runtime = 60 * 1  # in minutes


model_state, optimizer_state, config_sample = torch.load(os.path.join("models_diff", "prior_diff_real_checkpointkc_n_0_epoch_16.cpkt"))
# module_prefix = 'module.'
# model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
# for key in list(model_state.keys()):
#     model_state[key.replace( 'decoder','decoder_dict.standard')] = model_state.pop(key)
        
        
def train_function(config_sample,model_state, i, add_name=""):
    start_time = time.time()
    N_epochs_to_save = 50

    def save_callback(model, epoch):
        if not hasattr(model, "last_saved_epoch"):
            model.last_saved_epoch = 0
        if (
            (time.time() - start_time) / (maximum_runtime * 60 / N_epochs_to_save)
        ) > model.last_saved_epoch:
            print("Saving model..")
            config_sample["epoch_in_training"] = epoch
            save_model(
                model,
                base_path,
                f"models_diff/prior_diff_real_checkpoint{add_name}_n_{i}_epoch_{model.last_saved_epoch}.cpkt",
                config_sample,
            )
            model.last_saved_epoch = (
                model.last_saved_epoch + 1
            )  # TODO: Rename to checkpoint

    get_model(
        config_sample,
        device,
        should_train=True,
        verbose=1,
        state_dict=model_state,
        epoch_callback=save_callback,
    )

    return


from train_config import config_sample
train_function(config_sample, model_state,17,  add_name="kc")

