from keras.layers import Conv2D, \
                         MaxPooling2D, \
                         Dense, Flatten, \
                         Dropout, \
                         Embedding, \
                         LSTM, \
                         Activation, \
                         Input, \
                         Concatenate
                         

from keras.models import Sequential, Model
from keras.regularizers import l2
from keras import optimizers

from keras import backend as K


from keras.utils.vis_utils import plot_model

import yaml

#===============================================================================
# Rete con
# <input_sh> ingressi e
# <dl2_units_num> uscite
# la rete con piu' uscite trasforma l'uscita in categorie
# e pretende una attivazione softmax
#===============================================================================


def Net_f(filePar):
    # legge dal file di parametri il nome del file con la rete
    with open(filePar, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        # CARICO I PARAMETRI DELLA RETE
        input_shape = cfg["parAlg"]["input_shape"]

        dropout = cfg["parAlg"]["dropout"]


        kernel_num = cfg["parAlg"]["kernel_num"]
        kernel_size = cfg["parAlg"]["kernel_size"]

        dl1_units_num = cfg["parAlg"]["dl1_units_num"]


        dl2_units_num = cfg["parAlg"]["dl2_units_num"]


        learning_rate = cfg["parAlg"]["learning_rate"]


    return Net(input_shape, dropout, 

            kernel_num, kernel_size,

            dl1_units_num, 

            dl2_units_num, 

            learning_rate
            )


#-------------------------------------------------------------------------------
def Net(input_sh,

        dropout,

        kernel_num,
        kernel_size,

        dl1_units_num,

        dl2_units_num,

        learning_rate
        ):

    # input_C1_C2 = Input(shape = input_sh)

   

    model = Sequential()

    model.add(Conv2D(kernel_num, kernel_size, input_shape=input_sh))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers

    model.add(Dense(dl1_units_num, activation="relu"))
    model.add(Dropout(dropout))

    model.add(Dense(dl2_units_num,activation="softmax"))

    optim = optimizers.Adam(lr=learning_rate)

    model.compile(optimizer='adam',  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def visualizza(model, nomeFile):
    plot_model(model, to_file=nomeFile)
