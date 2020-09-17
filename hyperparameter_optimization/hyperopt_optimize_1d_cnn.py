
"""Auto-optimizing a neural network with Hyperopt (TPE algorithm)."""


from model_config_1d_cnn import CNN_Config_1d

from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from hyperopt import hp, tpe, fmin, Trials
from hyperopt import STATUS_OK, STATUS_FAIL
import pickle
import os
import traceback
import numpy as np
import pandas as pd




space = {
    # This loguniform scale will multiply the learning rate, so as to make
    # it vary exponentially, in a multiplicative fashion rather than in
    # a linear fashion, to handle his exponentialy varying nature:
    'lr_rate_mult': hp.loguniform('lr_rate_mult', -0.5, 0.5),
    # L2 weight decay:
    'l2_weight_reg_mult': hp.loguniform('l2_weight_reg_mult', -1.3, 1.3),
    # Batch size fed for each gradient update
    'batch_size': hp.quniform('batch_size', 80, 180, 10),
    # Number of EPOCHS
    'epochs': hp.quniform('epochs', 10, 60, 10),
    # Choice of optimizer:
    'optimizer': hp.choice('optimizer', ['Adam', 'Nadam', 'RMSprop']),
    # Uniform distribution in finding appropriate dropout values, conv layers
    'conv_dropout_drop_proba': hp.uniform('conv_dropout_proba', 0.0, 0.35),
    # Uniform distribution in finding appropriate dropout values, FC layers
    'fc_dropout_drop_proba': hp.uniform('fc_dropout_proba', 0.0, 0.6),
    # Uniform distribution in finding appropriate dropout values, lstm nb_conv_pool_layers
    'lstm_dropout_drop_proba': hp.uniform('lstm_dropout_drop_proba', 0.0, 0.5),
    # Use batch normalisation at more places?
    'use_BN': hp.choice('use_BN', [False, True]),
    # Use different Architectures
    'arch_type': hp.choice('arch_type', ['vgg', 'normal']),
    # Starting conv+pool layer for residual connections:
    'conv_pool_res_start_idx': hp.quniform('conv_pool_res_start_idx', 0, 2, 1),
    # Use residual connections? If so, how many more to stack?
    'residual': hp.choice(
        'residual', [None, hp.quniform(
            'residual_units', 1 - 0.499, 4 + 0.499, 1)]
    ),
    # Let's multiply the "default" number of hidden units:
    'conv_hiddn_units_mult': hp.loguniform('conv_hiddn_units_mult', -0.6, 0.6),
    # Number of conv+pool layers stacked:
    'nb_conv_pool_layers': hp.quniform('nb_conv_pool_layers', 2, 5, 1),
    # No of convolutions to stack on each layer for vgg networks
    'no_stack_vgg': hp.choice('no_stack_vgg', [2, 3, 4]),
    # The kernel_size for convolutions:
    'conv_kernel_size': 3.0,
    # The kernel_size for residual convolutions:
    'res_conv_kernel_size': 3.0,
    # Maxpool kernel size
    'pool_size': hp.choice('pool_size', [2, 4]),
    # No. of lstm cells
    'lstm_layer':  hp.choice('lstm_layer', [None, hp.loguniform('lstm_units_mult', -1.0, 1.0)]),
    # Amount of fully-connected units after convolution feature map
    'fc_layer': hp.choice('fc_layer', [None, hp.loguniform('fc_units_mult', -0.6, 0.6)]),
    # Use one more lstm layers
    'one_more_lstm_layer': hp.choice('one_more_lstm_layer', [True, False]),
    # Use one more fc layers
    'one_more_fc_layer': hp.choice('one_more_fc_layer', [True, False]),
    # Activations that are used everywhere
    'activation': 'relu'
}

train_id = pd.read_csv("data/X_train_10.csv", header=None).values.flatten()
test_id = pd.read_csv("data/X_test_10.csv", header=None).values.flatten()
Y = np.load("data/Y_RT_10.npy")
Y = to_categorical(Y)

utils = {
    'tensorboard_dir': "TensorBoard/ERP_RT_10",
    'weights_dir': 'models/ERP_RT_10',
    'nb_channels': 64,
    'nb_classes': 3,
    'nb_time_steps': 500,
    'data_path': "data/RT_10_ERPs/",
    'train_id': train_id,
    'test_id' : test_id,
    'Y': Y,
    'results_dir': "results/ERP_RT_10/",
    'hyperspace': space

}

cnn_config = CNN_Config_1d(utils, datatype=1)

def plot(hyperspace, file_name_prefix):
    """Plot a model from it's hyperspace."""
    model = cnn_config.build_model(hyperspace)
    plot_model(
        model,
        to_file='{}.png'.format(file_name_prefix),
        show_shapes=True
    )
    print("Saved model visualization to {}.png.".format(file_name_prefix))
    K.clear_session()
    del model


def plot_best_model():
    """Plot the best model found yet."""
    space_best_model = cnn_config.helper.load_best_hyperspace()
    if space_best_model is None:
        print("No best model to plot. Continuing...")
        return

    print("Best hyperspace yet:")
    cnn_config.helper.print_json(space_best_model)
    plot(space_best_model, "model_best_1d_cnn")


def optimize_cnn(hype_space):
    """Build a convolutional neural network and train it."""
    try:
        model, model_name, result, log_path = cnn_config.build_and_train(hype_space, save_best_weights=True, log_for_tensorboard=True)

        # Save training results to disks with unique filenames
        cnn_config.helper.save_json_result(model_name, result)

        K.clear_session()
        del model

        return result

    except Exception as err:
        try:
            K.clear_session()
        except:
            pass
        err_str = str(err)
        print(err_str)
        traceback_str = str(traceback.format_exc())
        print(traceback_str)
        return {
            'status': STATUS_FAIL,
            'err': err_str,
            'traceback': traceback_str
        }

    print("\n\n")


def run_a_trial():
    """Run one TPE meta optimisation step and save its results."""
    max_evals = nb_evals = 1

    print("Attempt to resume a past training if it exists:")

    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open("results_RT_ERP_10.pkl", "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        print("Starting from scratch: new trials.")

    best = fmin(
        optimize_cnn,
        space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    pickle.dump(trials, open("results_RT_ERP_10.pkl", "wb"))

    print("\nOPTIMIZATION STEP COMPLETE.\n")


if __name__ == "__main__":
    """Plot the model and run the optimisation forever (and saves results)."""


    print("Now, we train many models, one after the other. "
          "Note that hyperopt has support for cloud "
          "distributed training using MongoDB.")

    print("\nYour results will be saved in the folder named 'results/'. "
          "You can sort that alphabetically and take the greatest one. "
          "As you run the optimization, results are consinuously saved into a "
          "'results.pkl' file, too. Re-running optimize.py will resume "
          "the meta-optimization.\n")
    i = 0
    while True:

        # Optimize a new model with the TPE Algorithm:
        print("OPTIMIZING NEW MODEL:")
        try:
            run_a_trial()
        except Exception as err:
            err_str = str(err)
            print(err_str)
            traceback_str = str(traceback.format_exc())
            print(traceback_str)

        # Replot best model since it may have changed:
        print("PLOTTING BEST MODEL:")
        plot_best_model()
        # i += 1
