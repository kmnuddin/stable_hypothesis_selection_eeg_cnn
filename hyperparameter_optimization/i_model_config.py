from abc import ABC, abstractmethod
from utils.helper import Helper
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop


class ICNN_Config(ABC):

    @abstractmethod
    def __init__(self, utils, datatype=0):
        self.utils = utils
        self.tensorboard_dir = utils['tensorboard_dir']
        self.weights_dir = utils['weights_dir']
        self.nb_channels = utils['nb_channels']
        self.results_dir = utils['results_dir']
        self.hyper_space = utils['hyperspace']
        self.nb_classes = utils['nb_classes']
        self.starting_l2_reg = 0.0007
        self.optimizer_str_to_class = {
                'Adam': Adam,
                'Nadam': Nadam,
                'RMSprop': RMSprop
            }
        if datatype == 0:
            self.image_border_length = utils['image_border_length']
            self.train_dir = utils['train_dir']
            self.test_dir = utils['test_dir']
            self.helper = Helper(self.train_dir, self.test_dir, self.results_dir)
        elif datatype == 1:
            self.nb_time_steps = utils['nb_time_steps']
            self.data_path = utils['data_path']
            self.train_id = utils['train_id']
            self.test_id = utils['test_id']
            self.Y = utils['Y']
            self.helper = Helper(None, None, self.results_dir)



    @abstractmethod
    def build_and_train(self, space, save_best_weights=False, log_for_tensorboard=False, train=True):
        pass

    @abstractmethod
    def build_model(self, space):
        pass

    @abstractmethod
    def random_image_mirror_left_right(self, input_layer):
        pass

    @abstractmethod
    def bn(self, prev_layer):
        pass

    @abstractmethod
    def dropout(self, prev_layer, space, for_convolution_else_fc=True):
        pass

    @abstractmethod
    def convolution(self, prev_layer, n_filters, space, force_ksize=None):
        pass

    @abstractmethod
    def residual(self, prev_layer, n_filters, space):
        pass

    @abstractmethod
    def auto_choose_pooling(self, prev_layer, n_filters, space):
        pass

    @abstractmethod
    def convolution_pooling(self, prev_layer, n_filters, space):
        pass

    @abstractmethod
    def inception_reduction(self, prev_layer, n_filters, space):
        pass
