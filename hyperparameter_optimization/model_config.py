
__author__ = "Guillaume Chevalier"
__copyright__ = "Copyright 2017, Guillaume Chevalier"
__license__ = "MIT License"
__notice__ = (
    "Some further edits by Guillaume Chevalier are made on "
    "behalf of Vooban Inc. and belongs to Vooban Inc. ")
# See: https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100/blob/master/LICENSE"

from i_model_config import ICNN_Config
from tensorflow import keras
from tensorflow.keras.datasets import cifar100  # from keras.datasets import cifar10
import tensorflow.keras.backend as K
import tensorflow as tf
from hyperopt import STATUS_OK, STATUS_FAIL
from utils.helper import Helper
from decimal import Decimal
import numpy as np
import uuid
import traceback
import os

class CNN_Config(ICNN_Config):


    def __init__(self, utils):
        super().__init__(utils)


    def build_and_train(self, space, save_best_weights=False, log_for_tensorboard=False, train=True):
        """Build the deep CNN model and train it."""
        K.set_learning_phase(1)
        K.set_image_data_format('channels_last')
        train_it, test_it = self.helper.construct_data_generator(batch_size=int(space['batch_size']), target_size=(128,128), shuffle=True)

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

        input_layer = keras.layers.Input(
            (self.image_border_length, self.image_border_length, self.nb_channels))

        # current_layer = random_image_mirror_left_right(input_layer)

        if space['first_conv'] is not None:
            k = space['first_conv']
            current_layer = keras.layers.Conv2D(
                filters=16, kernel_size=(k, k), strides=(1, 1),
                padding='same', activation=space['activation'],
                kernel_regularizer=keras.regularizers.l2(
                    self.starting_l2_reg * space['l2_weight_reg_mult'])
            )(input_layer)
        else:
            current_layer = input_layer
        # Core loop that stacks multiple conv+pool layers, with maybe some
        # residual connections and other fluffs:
        n_filters = int(40 * space['conv_hiddn_units_mult'])
        for i in range(int(space['nb_conv_pool_layers'])):
            print(i)
            print(n_filters)
            print(current_layer.shape)

            current_layer = self.convolution(current_layer, n_filters, space)
            if space['use_BN']:
                current_layer = self.bn(current_layer)
            print(current_layer.shape)

            deep_enough_for_res = space['conv_pool_res_start_idx']
            if i >= deep_enough_for_res and space['residual'] is not None:
                current_layer = self.residual(current_layer, n_filters, space)
                print(current_layer.shape)

            current_layer = self.auto_choose_pooling(
                current_layer, n_filters, space)
            print(current_layer.shape)

            if space['nb_conv_pool_layers'] >= 4:
                n_filters += int(40 * space['conv_hiddn_units_mult'])
            else:
                n_filters *= 2

        # Fully Connected (FC) part:
        current_layer = keras.layers.Flatten()(current_layer)
        print(current_layer.shape)

        current_layer = keras.layers.Dense(
            units=int(1000 * space['fc_units_1_mult']),
            activation=space['activation'],
            kernel_regularizer=keras.regularizers.l2(
                self.starting_l2_reg * space['l2_weight_reg_mult'])
        )(current_layer)
        print(current_layer.shape)

        current_layer = self.dropout(
            current_layer, space, for_convolution_else_fc=False)

        if space['one_more_fc'] is not None:
            current_layer = keras.layers.Dense(
                units=int(750 * space['one_more_fc']),
                activation=space['activation'],
                kernel_regularizer=keras.regularizers.l2(
                    self.starting_l2_reg * space['l2_weight_reg_mult'])
            )(current_layer)
            print(current_layer.shape)

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
                lr=1e-5 * space['lr_rate_mult']
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


    def random_image_mirror_left_right(self, input_layer):
        """
        Flip each image left-right like in a mirror, randomly, even at test-time.

        This acts as a data augmentation technique. See:
        https://stackoverflow.com/questions/39574999/tensorflow-tf-image-functions-on-an-image-batch
        """
        return keras.layers.Lambda(function=lambda batch_imgs: tf.map_fn(
            lambda img: tf.image.random_flip_left_right(img), batch_imgs
        )
        )(input_layer)


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
        if force_ksize is not None:
            k = force_ksize
        else:
            k = int(round(space['conv_kernel_size']))
        return keras.layers.Conv2D(
            filters=n_filters, kernel_size=(k, k), strides=(1, 1),
            padding='same', activation=space['activation'],
            kernel_regularizer=keras.regularizers.l2(
                self.starting_l2_reg * space['l2_weight_reg_mult'])
        )(prev_layer)


    def residual(self, prev_layer, n_filters, space):
        """Some sort of residual layer, parametrized by the space."""
        current_layer = prev_layer
        for i in range(int(round(space['residual']))):
            lin_current_layer = keras.layers.Conv2D(
                filters=n_filters, kernel_size=(1, 1), strides=(1, 1),
                padding='same', activation='linear',
                kernel_regularizer=keras.regularizers.l2(
                    self.starting_l2_reg * space['l2_weight_reg_mult'])
            )(current_layer)

            layer_to_add = self.dropout(current_layer, space)
            layer_to_add = self.convolution(
                layer_to_add, n_filters, space,
                force_ksize=int(round(space['res_conv_kernel_size'])))

            current_layer = keras.layers.add([
                lin_current_layer,
                layer_to_add
            ])
            if space['use_BN']:
                current_layer = self.bn(current_layer)
        if not space['use_BN']:
            current_layer = self.bn(current_layer)

        return self.bn(current_layer)


    def auto_choose_pooling(self, prev_layer, n_filters, space):
        """Deal with pooling in convolution steps."""
        if space['pooling_type'] == 'all_conv':
            current_layer = self.convolution_pooling(
                prev_layer, n_filters, space)

        elif space['pooling_type'] == 'inception':
            current_layer = self.inception_reduction(prev_layer, n_filters, space)

        elif space['pooling_type'] == 'avg':
            current_layer = keras.layers.AveragePooling2D(
                pool_size=(2, 2)
            )(prev_layer)

        else:  # 'max'
            current_layer = keras.layers.MaxPooling2D(
                pool_size=(2, 2)
            )(prev_layer)

        return current_layer


    def convolution_pooling(self, prev_layer, n_filters, space):
        """
        Pooling with a convolution of stride 2.

        See: https://arxiv.org/pdf/1412.6806.pdf
        """
        current_layer = keras.layers.Conv2D(
            filters=n_filters, kernel_size=(3, 3), strides=(2, 2),
            padding='same', activation='linear',
            kernel_regularizer=keras.regularizers.l2(
                self.starting_l2_reg * space['l2_weight_reg_mult'])
        )(prev_layer)

        if space['use_BN']:
            current_layer = self.bn(current_layer)

        return current_layer


    def inception_reduction(self, prev_layer, n_filters, space):
        """
        Reduction block, vaguely inspired from inception.

        See: https://arxiv.org/pdf/1602.07261.pdf
        """
        n_filters_a = int(n_filters * 0.33 + 1)
        n_filters = int(n_filters * 0.4 + 1)

        conv1 = self.convolution(prev_layer, n_filters_a, space, force_ksize=3)
        conv1 = self.convolution_pooling(prev_layer, n_filters, space)

        conv2 = self.convolution(prev_layer, n_filters_a, space, 1)
        conv2 = self.convolution(conv2, n_filters, space, 3)
        conv2 = self.convolution_pooling(conv2, n_filters, space)

        conv3 = self.convolution(prev_layer, n_filters, space, force_ksize=1)
        conv3 = keras.layers.MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), padding='same'
        )(conv3)

        current_layer = keras.layers.concatenate([conv1, conv2, conv3], axis=-1)

        return current_layer
