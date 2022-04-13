from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from keras.optimizers import Adam, Nadam,RMSprop
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras import regularizers

def create_model(n_features=60,n_labels=1):
    neurons_l = [80]
    activation=['sigmoid','tanh','sigmoid','sigmoid']
    n_layers = 1
    init_mode = ['uniform','normal','normal','normal']
    optimizer = np.random.choice([Nadam(lr=0.001),RMSprop(lr=0.001)])
    model = Sequential()
    for n in range(n_layers):
        if n==0:
            model.add(Dense(np.random.choice(np.arange(neurons_l[n]-10,neurons_l[n]+20,10)), 
                            input_shape=(n_features,),
                            kernel_initializer=init_mode[n],
#                            kernel_regularizer=regularizers.l2(0.002), 
                            activation=activation[n]
                            ))
        else:
            model.add(Dense(neurons_l[n], 
                            kernel_initializer=init_mode[n], 
                            activation=activation[n],
                            kernel_regularizer=regularizers.l2(0.3),
                            ))
    model.add(Dense(n_labels,activation='linear',
                    kernel_regularizer=regularizers.l2(0.2),
                    ))

    model.compile(optimizer=optimizer,
                  loss='mean_squared_error')
    return model


def predict(xt,xs,yt):
    K.clear_session()
    epochs = 5000
    callback_patience = 40
    callbacks  = [EarlyStopping(monitor='val_loss',
                                patience=callback_patience,
                                verbose=1)]
    nnmodel = create_model(n_features=xt.shape[1],n_labels=1)
    batch_size = 64
    print('Proceeding to model fit')
    _ = nnmodel.fit(xt,yt,
                    callbacks=callbacks,
                    validation_split=0.1,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=2)
    print('finished model fits')
    return(nnmodel.predict(xs))

