from i_model_config import ICNN_Config

from tensorflow import keras
import tensorflow.keras.backend as K

from utils.custom_generator import DataGenerator

from hyperopt import STATUS_OK, STATUS_FAIL
from decimal import Decimal
import numpy as np
import uuid
import traceback
import os

class CNN_Config_1d(ICNN_Config):

    def __init__(self, utils, datatype):
        super().__init__(utils, datatype)


    def build_and_train(self, space, save_best_weights=False, log_for_tensorboard=False, train=True):
        """Build the deep CNN model and train it."""
        K.set_learning_phase(1)
        K.set_image_data_format('channels_last')
        train_it = DataGenerator(self.data_path, self.train_id, self.Y, target_shape=(self.nb_time_steps, self.nb_channels),
                                      num_classes=self.nb_classes, batch_size=int(space['batch_size']))
        test_it = DataGenerator(self.data_path, self.test_id, self.Y, target_shape=(self.nb_time_steps, self.nb_channels), num_classes=self.nb_classes,
                                      batch_size=int(space['batch_size']))

        # if log_for_tensorboard:
        #     # We need a smaller batch size to not blow memory with tensorboard
        #     space["lr_rate_mult"] = space["lr_rate_mult"] / 10.0
        #     space["batch_size"] = space["batch_size"] / 10.0

        model = self.build_model(space)

        # K.set_learning_phase(1)

        model_uuid = str(uuid.uuid4())[:5]

        callbacks = []

        # Weight saving callback:
        if save_best_weights:
            weights_save_path = os.path.join(
                self.weights_dir , '{}.h5'.format(model_uuid))
            print("Model's weights will be saved to: {}".format(weights_save_path))
            if not os.path.exists(self.weights_dir ):
                os.makedirs(self.weights_dir )

            callbacks.append(keras.callbacks.ModelCheckpoint(
                weights_save_path,
                monitor='val_accuracy',
                save_best_only=True, mode='max'))

        # TensorBoard logging callback:
        log_path = None
        if log_for_tensorboard:
            log_path = os.path.join(self.tensorboard_dir, model_uuid)
            print("Tensorboard log files will be saved to: {}".format(log_path))
            if not os.path.exists(log_path):
                os.makedirs(log_path)

            # Right now Keras's TensorBoard callback and TensorBoard itself are not
            # properly documented so we do not save embeddings (e.g.: for T-SNE).

            # embeddings_metadata = {
            #     # Dense layers only:
            #     l.name: "../10000_test_classes_labels_on_1_row_in_plain_text.tsv"
            #     for l in model.layers if 'dense' in l.name.lower()
            # }

            tb_callback = keras.callbacks.TensorBoard(
                log_dir=log_path,
                histogram_freq=2,
                # write_images=True, # Enabling this line would require more than 5 GB at each `histogram_freq` epoch.
                write_graph=True
                # embeddings_freq=3,
                # embeddings_layer_names=list(embeddings_metadata.keys()),
                # embeddings_metadata=embeddings_metadata
            )
            tb_callback.set_model(model)
            callbacks.append(tb_callback)

        redonplat = keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max", patience=5, verbose=2, min_lr=1e-7)
        callbacks.append(redonplat)

        if not train:
            return model, callbacks

        # Train net:
        history = model.fit_generator(
            train_it,
            epochs=int(space['epochs']),
            verbose=1,
            callbacks=callbacks,
            validation_data=test_it
        ).history

        # # Test net:
        K.set_learning_phase(0)
        score = model.evaluate(test_it, verbose=0)
        max_acc = max(history['val_accuracy'])
        #
        model_name = "model_{:.2f}_id_{}".format(round(max_acc, 2), model_uuid)
        print("Model name: {}".format(model_name))
        print(max_acc)
        # Note: to restore the model, you'll need to have a keras callback to
        # save the best weights and not the final weights. Only the result is
        # saved.
        print(history.keys())
        print(history)
        print(score)
        result = {
            # We plug "-val_fine_outputs_acc" as a
            # minimizing metric named 'loss' by Hyperopt.
            'loss': -max_acc.astype(np.float64),
            'real_loss': score[0].astype(np.float64),
            # Fine stats:
            'best_val_loss': min(history['val_loss']).astype(np.float64),
            'best_val_accuracy': max(history['val_accuracy']).astype(np.float64),
            # Misc:
            'model_name': model_name,
            'space': space,
            # 'history': history,
            'status': STATUS_OK
        }

        print("RESULT:")
        print(result)
        self.helper.print_json(result)

        return model, model_name, result, log_path

    def build_model(self, space):
        """Create model according to the hyperparameter space given."""
        print("Hyperspace:")
        print(space)

        input_layer = keras.layers.Input((self.nb_time_steps, self.nb_channels))

        # current_layer = random_image_mirror_left_right(input_layer)


        current_layer = input_layer
        # Core loop that stacks multiple conv+pool layers, with maybe some
        # residual connections and vgg type layers
        n_filters = int(20 * space['conv_hiddn_units_mult'])
        for i in range(int(space['nb_conv_pool_layers'])):

            if space['arch_type'] == 'normal':
                current_layer = self.convolution(current_layer, n_filters, space)
                if space['use_BN']:
                    current_layer = self.bn(current_layer)
                    current_layer = keras.layers.Activation(space['activation'])(current_layer)

                deep_enough_for_res = space['conv_pool_res_start_idx']
                if i >= deep_enough_for_res and space['residual'] is not None:
                    current_layer = self.residual(current_layer, n_filters, space)



                if space['nb_conv_pool_layers'] >= 4:
                    n_filters += int(20 * space['conv_hiddn_units_mult'])
                else:
                    n_filters *= 2

            elif space['arch_type'] == 'vgg':
                if i == 0:
                    force_ksize = int(space['conv_kernel_size'] + 2)
                else:
                    force_ksize = None
                for j in range(space['no_stack_vgg']):
                    current_layer = self.convolution(current_layer, n_filters, space, force_ksize)

                    if space['use_BN']:
                        current_layer = self.bn(current_layer)
                        current_layer = keras.layers.Activation(space['activation'])(current_layer)

                if i % 2 == 0:
                    n_filters *= 2


            if i == space['nb_conv_pool_layers'] - 1:
                if space['lstm_layer'] is None:
                    current_layer = keras.layers.MaxPooling1D(pool_size=space['pool_size'])(current_layer)
                    current_layer = keras.layers.Flatten()(current_layer)
                    continue

            current_layer = keras.layers.MaxPooling1D(pool_size=space['pool_size'])(current_layer)


        ## LSTM connections

        if space['lstm_layer'] is not None:

            current_layer = keras.layers.Bidirectional(keras.layers.LSTM(int(200 * space['lstm_layer']), return_sequences=True))(current_layer)
            current_layer = keras.layers.Dropout(space['lstm_dropout_drop_proba'])(current_layer)

            if space['one_more_lstm_layer']:
                current_layer = keras.layers.Bidirectional(keras.layers.LSTM(int(200 * space['lstm_layer']), return_sequences=True))(current_layer)
                current_layer = keras.layers.Dropout(space['lstm_dropout_drop_proba'])(current_layer)

            current_layer = keras.layers.Flatten()(current_layer)

        # Fully Connected (FC) part:

        if space['fc_layer']:
            current_layer = keras.layers.Dense(
                units=int(200 * space['fc_layer']),
                activation=space['activation'],
                kernel_regularizer=keras.regularizers.l2(
                    self.starting_l2_reg * space['l2_weight_reg_mult'])
            )(current_layer)

            current_layer = self.dropout(
                current_layer, space, for_convolution_else_fc=False)

            if space['one_more_fc_layer']:
                current_layer = keras.layers.Dense(
                    units=int(100 * space['fc_layer']),
                    activation=space['activation'],
                    kernel_regularizer=keras.regularizers.l2(
                        self.starting_l2_reg * space['l2_weight_reg_mult'])
                )(current_layer)

                current_layer = self.dropout(
                    current_layer, space, for_convolution_else_fc=False)

        fine_outputs = keras.layers.Dense(
            units=self.nb_classes,
            activation="softmax",
            kernel_regularizer=keras.regularizers.l2(
                self.starting_l2_reg * space['l2_weight_reg_mult']),
        )(current_layer)

        # Finalize model:
        model = keras.models.Model(
            inputs=[input_layer],
            outputs=[fine_outputs]
        )
        model.compile(
            optimizer=self.optimizer_str_to_class[space['optimizer']](
                lr=1e-3 * space['lr_rate_mult']
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def bn(self, prev_layer):
        """Perform batch normalisation."""
        return keras.layers.BatchNormalization()(prev_layer)


    def dropout(self, prev_layer, space, for_convolution_else_fc=True):
        """Add dropout after a layer."""
        if for_convolution_else_fc:
            return keras.layers.Dropout(
                rate=space['conv_dropout_drop_proba']
            )(prev_layer)
        else:
            return keras.layers.Dropout(
                rate=space['fc_dropout_drop_proba']
            )(prev_layer)


    def convolution(self, prev_layer, n_filters, space, force_ksize=None):
        """Basic convolution layer, parametrized by the space."""
        if space['use_BN']:
            activation = None
        else:
            activation = space['activation']
        if force_ksize is not None:
            k = force_ksize
        else:
            k = int(round(space['conv_kernel_size']))
        return keras.layers.Conv1D(
            filters=n_filters, kernel_size=k, activation=activation,
            padding='valid', kernel_regularizer=keras.regularizers.l2( self.starting_l2_reg * space['l2_weight_reg_mult'])
        )(prev_layer)


    def residual(self, prev_layer, n_filters, space):
        """Some sort of residual layer, parametrized by the space."""
        current_layer = prev_layer
        for i in range(int(round(space['residual']))):
            lin_current_layer = keras.layers.Conv1D(
                filters=n_filters, kernel_size=1, padding='valid', activation='linear',
                kernel_regularizer=keras.regularizers.l2(
                    self.starting_l2_reg * space['l2_weight_reg_mult'])
            )(current_layer)

            layer_to_add = self.dropout(current_layer, space)
            layer_to_add = self.convolution(
                layer_to_add, n_filters, space,
                force_ksize=int(round(space['res_conv_kernel_size'])))
            if space['use_BN']:
                layer_to_add = keras.layers.Activation(space['activation'])(layer_to_add)

            current_layer = keras.layers.add([
                lin_current_layer,
                layer_to_add
            ])
            if space['use_BN']:
                current_layer = self.bn(current_layer)
        if not space['use_BN']:
            current_layer = self.bn(current_layer)

        return self.bn(current_layer)

    def random_image_mirror_left_right(self, input_layer):
        raise NotImplementedError

    def auto_choose_pooling(self, prev_layer, n_filters, space):
        raise NotImplementedError

    def inception_reduction(self, prev_layer, n_filters, space):
        raise NotImplementedError

    def convolution_pooling(self, prev_layer, n_filters, space):
        raise NotImplementedError
