import os
from typing import Union
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from sklearn.utils import class_weight

class SiameseTripletModel():
    
    img_size = (224, 224)

    def __init__(self, 
                 out_folder: Union[str, None] = None, # model weight output folder
                 model: dict={}, # model parameters
                 train: dict={} # training parameters
                 ) -> None:
        self._out_folder = out_folder
        self.model = None
        self._train = train
        self._base_layer_count: int = 0
        self._initialize_model(**model)
        return
    
    def _initialize_model(self, 
                          encoder_name: str='efficientnetv2b0',
                          training: bool=False, # training mode
                          dense_units: int=512,
                          bn: bool=False,
                          top_dropout_rate: float=0.2,
                          weights: str='',
                          ) -> None:
        '''
        encoder_name: encoder name, 'vgg19' or 'efficientnetv2b0'
        freeze_until: freeze layers until this layer, '' means no freeze, 'all' means freeze all layers
        dense_units: number of units in the dense layer
        top_dropout_rate: dropout rate
        weights: path to existing weights to load, '': use 'imagenet'
        '''
        if encoder_name == 'vgg19':
            base = tf.keras.applications.VGG19(include_top=False, 
                weights='imagenet', 
                input_shape=(*SiameseTripletModel.img_size, 3),
                pooling='max')
            # name the input layer
            base.layers[0]._name = 'vgg19_input'
            base.layers[-1]._name = 'vgg19_max_pooling2d'
            # 512 after max pooling
        elif encoder_name == 'efficientnetv2b0':
            base = tf.keras.applications.EfficientNetV2B0(include_top=False,
                input_shape=(*SiameseTripletModel.img_size, 3),
                weights="imagenet",
                include_preprocessing=True,
                pooling="avg")
            base.layers[0]._name = 'efficientnetv2b0_input'
            base.layers[-1]._name = 'efficientnetv2b0_avg_pooling2d'
            # 1280 after avg pooling
        else:
            raise NotImplementedError(f'Encoder {encoder_name} not implemented')
        # https://pub.towardsai.net/batchnorm-for-transfer-learning-df17d2897db6
        input = tf.keras.layers.Input(shape=(*SiameseTripletModel.img_size, 3))
        x = base.call(input, training=training) 
        self._base_layer_count = len(base.layers)
        if bn == True:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(rate=top_dropout_rate,
                                    name='top_dropout')(x)
        if dense_units != 0:
            # add fully connected layer
            x = tf.keras.layers.Dense(units=dense_units, 
                                      # kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                      activation='relu', 
                                      name='top_dense')(x)
        norm_layer = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), 
                                            name='norm_layer')(x)
        model = tf.keras.Model(inputs=input, 
                                outputs=norm_layer, 
                                name=encoder_name)
        if weights != '':
            model.load_weights(weights)
        self.model = model
        return

    def _compile(self, optimizer: dict={}, 
                 loss: dict={},
                 freeze_until: str='',
                 freeze_bn: Union[bool, None]=None) -> None:
        '''
        freeze_bn: freeze batch normalization layers, null: overwritten by freeze_until
        '''
        model = self.model
        # unfreeze all layers
        for l in model.layers:
            l.trainable = True
        # freeze layers
        if freeze_until != '':
            for l in model.layers[:self._base_layer_count]:
                if l.name == freeze_until:
                    break
                l.trainable = False
        # freeze batch normalization layers
        if freeze_bn is not None:
            for l in model.layers[:self._base_layer_count]:
                if isinstance(l, tf.keras.layers.BatchNormalization):
                    l.trainable = not freeze_bn
                    # l.momentum = 0.9
        # optimizer
        optimizer_name = optimizer['name']
        if optimizer_name == 'SGD':
            lr = optimizer['learning_rate']
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr, 
                                                momentum=0.9, nesterov=True)
        elif optimizer_name == 'Adam':
            lr = optimizer['learning_rate']
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            raise NotImplementedError(f'Optimizer {optimizer_name} not implemented')
        # loss
        loss_name = loss['name']
        if loss_name == 'tripletsemihard':
            margin = loss['margin']
            loss = tfa.losses.TripletSemiHardLoss(margin=margin)
        elif loss_name == 'triplethard':
            margin = loss['margin']
            loss = tfa.losses.TripletHardLoss(margin=margin)
        else:
            raise NotImplementedError(f'Loss {loss_name} not implemented')
        # compile
        model.compile(optimizer=optimizer,
            loss=loss,
            metrics=['accuracy'])
        return
    
    def _fit(self, train_generator, val_generator, 
             epoch: int=1, # additional epochs
             batch_size: int=32, 
             initial_epoch: int=0) -> tf.keras.callbacks.History: 
        lr_red = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', 
                                patience = 4,
                                verbose = 1,
                                mode = 'min',
                                factor = 0.1, 
                                min_lr = 1e-7)
        es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', 
                                mode = 'min', 
                                verbose = 1, 
                                patience = 5)
        skip_save = self._out_folder is None
        callbacks = [lr_red, es]
        if skip_save == False:
            weights_file = os.path.join(self._out_folder, r'weights_e{epoch:02d}_best.h5')
            mc = tf.keras.callbacks.ModelCheckpoint(weights_file,
                                verbose = 1,
                                monitor = 'val_loss', 
                                mode = 'min', 
                                save_best_only = True, 
                                save_weights_only = True)
            log_file = os.path.join(self._out_folder, 'training.log')
            csv = tf.keras.callbacks.CSVLogger(log_file, append=True)
            callbacks.append(mc)
            callbacks.append(csv)
        model = self.model
        history = model.fit(train_generator, 
            epochs=epoch + initial_epoch,
            initial_epoch=initial_epoch,
            verbose=1, 
            batch_size=batch_size, 
            validation_data=val_generator,
            validation_freq=1, 
            callbacks=callbacks)
        if skip_save == False:
            weights_file = os.path.join(self._out_folder, 
                                        'weights_e{:02d}_final.h5'.format(history.epoch[-1] + 1))
            model.save_weights(weights_file)
        return history

    def train(self, train_generator: tf.keras.utils.Sequence, 
              val_generator: tf.keras.utils.Sequence) -> None:
        train_opt = self._train
        if isinstance(train_opt, dict):
            train_opt = [train_opt]
        initial_epoch = 0
        for i, opt in enumerate(train_opt):
            if len(train_opt) > 1:
                print(f'Training stage {i+1}...')
            optimizer = opt['optimizer']
            loss = opt['loss']
            freeze_until = opt['freeze_until']
            self._compile(optimizer, loss, freeze_until)
            fit_opt = opt['fit']
            history = self._fit(train_generator, val_generator, 
                                initial_epoch=initial_epoch, **fit_opt)
            if len(history.epoch) == 0:
                print('No epochs trained')
                continue
            initial_epoch = history.epoch[-1] + 1
        return

class SupervisedModel():

    img_size = (224, 224)

    def __init__(self,
                out_folder: Union[str, None]=None, # model weight output folder
                model: dict={}, # model parameters
                train: dict={} # training parameters
                ) -> None:
        self._out_folder = out_folder
        self.model = None
        self._train = train
        self._base_layer_count: int = 0
        self._initialize_model(**model)
        return
    
    def _initialize_model(self, 
                          encoder_name: str='efficientnetv2b0', # encoder name
                          training: bool=False, # training mode
                          n_classes: int=2, # number of classes
                          dense_units: int=512, # number of units in the dense layer
                          bn: bool=False,
                          top_dropout_rate: float=0.2, # dropout rate
                          weights: str='' # path to existing weights to load, '': use 'imagenet'
                          ) -> None:
        if encoder_name == 'vgg19':
            raise NotImplementedError('VGG19 not implemented')
            base = tf.keras.applications.vgg19.VGG19(
                input_shape=(*SupervisedModel.img_size, 3),
                weights='imagenet', 
                include_top=False)
        elif encoder_name == 'efficientnetv2b0':
            base = tf.keras.applications.EfficientNetV2B0(include_top=False,
                input_shape=(*SupervisedModel.img_size, 3),
                weights="imagenet",
                include_preprocessing=True,
                pooling="avg")
            base.layers[0]._name = 'efficientnetv2b0_input'
            base.layers[-1]._name = 'efficientnetv2b0_avg_pooling2d'
            # 1280 after avg pooling
        else:
            raise NotImplementedError(f'Encoder {encoder_name} not implemented')
        input = tf.keras.layers.Input(shape=(*SupervisedModel.img_size, 3))
        x = base.call(input, training=training)
        self._base_layer_count = len(base.layers)
        if bn == True:
            x = tf.keras.layers.BatchNormalization(name='final_bn')(x)
        x = tf.keras.layers.Dropout(rate=top_dropout_rate,
                                    name='top_dropout')(x)
        if dense_units != 0:
            # add fully connected layer
            x = tf.keras.layers.Dense(units=dense_units,
                                        activation='relu',
                                        name='top_dense')(x)
        embedding = x
        prediction = tf.keras.layers.Dense(n_classes, 
                                            activation='softmax', 
                                            name='predictions')(x)
        model = tf.keras.Model(inputs=input, 
                                outputs=[prediction, embedding], 
                                name=encoder_name)
        if weights != '':
            model.load_weights(weights)
        self.model = model
        return
    
    def _compile(self, optimizer: dict={}, 
                 freeze_until: str='',
                 freeze_bn: Union[bool, None]=None) -> None:
        model = self.model
        # unfreeze all layers
        for l in model.layers:
            l.trainable = True
        # freeze layers in base
        if freeze_until != '':
            for l in model.layers[:self._base_layer_count]:
                if l.name == freeze_until:
                    break
                l.trainable = False
        # freeze batch normalization layers
        if freeze_bn is not None:
            for l in model.layers[:self._base_layer_count]:
                if isinstance(l, tf.keras.layers.BatchNormalization):
                    l.trainable = not freeze_bn
        # optimizer
        optimizer_name = optimizer['name']
        if optimizer_name == 'SGD':
            lr = optimizer['learning_rate']
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr, 
                                                momentum=0.9, nesterov=True)
        elif optimizer_name == 'Adam':
            lr = optimizer['learning_rate']
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            raise NotImplementedError(f'Optimizer {optimizer_name} not implemented')
        # loss
        loss = tf.keras.losses.sparse_categorical_crossentropy
        # compile
        model.compile(optimizer=optimizer,
            loss={'predictions': loss},
            metrics={'predictions': 'accuracy'})
        return
    
    def _fit(self, train_generator, val_generator, 
             epoch: int=1, # additional epochs
             batch_size: int=32, 
             initial_epoch: int=0) -> tf.keras.callbacks.History: 
        lr_red = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', 
                                patience = 4,
                                verbose = 1,
                                mode = 'min',
                                factor = 0.1, 
                                min_lr = 1e-7)
        es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', 
                   mode = 'min', 
                   verbose = 1, 
                   patience = 5)
        skip_save = self._out_folder is None
        callbacks = [lr_red, es]
        if skip_save == False:
            weights_file = os.path.join(self._out_folder, r'weights_e{epoch:02d}_best.h5')
            mc = tf.keras.callbacks.ModelCheckpoint(weights_file,
                                verbose = 1,
                                monitor = 'val_loss', 
                                mode = 'min', 
                                save_weights_only = True,
                                save_best_only = True)
            log_file = os.path.join(self._out_folder, 'training.log')
            csv = tf.keras.callbacks.CSVLogger(log_file, append=True)
            callbacks.append(mc)
            callbacks.append(csv)
        # calculate class weights
        y_true = train_generator._df['label'].cat.codes.values
        class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                        classes=np.unique(y_true),
                                                        y=y_true)
        class_weights = dict(enumerate(class_weights))
        model = self.model
        history = model.fit(train_generator, 
            epochs=epoch + initial_epoch,
            initial_epoch=initial_epoch,
            verbose=1, 
            batch_size=batch_size, 
            validation_data=val_generator,
            validation_freq=1, 
            class_weight = class_weights,
            callbacks=callbacks)
        if skip_save == False:
            weights_file = os.path.join(self._out_folder, 
                                        'weights_e{:02d}_final.h5'.format(history.epoch[-1] + 1))
            model.save_weights(weights_file)
        return history
    
    def train(self, train_generator: tf.keras.utils.Sequence,
              val_generator: tf.keras.utils.Sequence) -> None:
        train_opt = self._train
        if isinstance(train_opt, dict):
            train_opt = [train_opt]
        initial_epoch = 0
        for i, opt in enumerate(train_opt):
            if len(train_opt) > 1:
                print(f'Training stage {i+1}...')
            optimizer = opt['optimizer']
            freeze_until = opt['freeze_until']
            self._compile(optimizer, freeze_until)
            fit_opt = opt['fit']
            history = self._fit(train_generator, val_generator, 
                                initial_epoch=initial_epoch, **fit_opt)
            if len(history.epoch) == 0:
                print('No epochs trained')
                continue
            initial_epoch += history.epoch[-1] + 1
        return

def load_model(opt: dict) -> object:
    opt = opt.copy()
    name = opt['name']
    del opt['name']
    if name == 'siamese':
        model_class = SiameseTripletModel(**opt)
    elif name == 'supervised':
        model_class = SupervisedModel(**opt)
    else:
        raise NotImplementedError(f'name {name} not implemented')
    return model_class
