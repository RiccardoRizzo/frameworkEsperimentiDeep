from keras.layers import Conv1D, \
                         MaxPooling1D, \
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
        cfg = yaml.load(ymlfile)
        # CARICO I PARAMETRI DELLA RETE
        input_shape = cfg["parAlg"]["input_shape"]

        dropout = cfg["parAlg"]["dropout"]
        beta = cfg["parAlg"]["beta"]

        cn1_kernel_num = cfg["parAlg"]["cn1_kernel_num"]
        cn1_kernel_size = cfg["parAlg"]["cn1_kernel_size"]


        cn2_kernel_num = cfg["parAlg"]["cn2_kernel_num"]
        cn2_kernel_size = cfg["parAlg"]["cn2_kernel_size"]


        cn_activation = cfg["parAlg"]["cn_activation"]


        dl1_units_num = cfg["parAlg"]["dl1_units_num"]
        dl1_activation = cfg["parAlg"]["dl1_activation"]

        dl2_units_num = cfg["parAlg"]["dl2_units_num"]
        dl2_activation = cfg["parAlg"]["dl2_activation"]

        learning_rate = cfg["parAlg"]["learning_rate"]


    return Net(input_shape, dropout, beta,

            cn1_kernel_num, cn1_kernel_size,

            cn2_kernel_num, cn2_kernel_size,

            cn_activation,

            dl1_units_num,  dl1_activation,

            dl2_units_num,  dl2_activation,

            learning_rate
            )


#-------------------------------------------------------------------------------
def Net(input_sh,

        dropout,
        beta,

        cn1_kernel_num,
        cn1_kernel_size,

        cn2_kernel_num,
        cn2_kernel_size,

        cn_activation,

        dl1_units_num,
        dl1_activation,

        dl2_units_num,
        dl2_activation,

        learning_rate
        ):

    input_C1_C2 = Input(shape = input_sh)

    #...............................

    C1_c= Conv1D(filters=cn1_kernel_num, kernel_size=cn1_kernel_size,
                 input_shape=input_sh,  kernel_regularizer=l2(beta),
                 padding='same')(input_C1_C2)
    C1_a = Activation(cn_activation)(C1_c)

    # il max pooling e' fatto usando tutto il vettore di ingresso
    # cosi' se e' presente la sequenza corrispondente ad un kernel
    # si accende l'outout
    # essendo k=3 abbiamo 64 configurazioni possibili, quindi
    # il numero di kernel puo' essere di poche decine
    # il numero di pesi da calcolare diventa molto piccolo
    # perche' ce' una uscita (che dovrebbe essere "binaria") per ogni kernel
    #sh= K.int_shape(C1_a)
    #C1_p = MaxPooling1D(pool_size=sh[1])(C1_a)

    C1_p = MaxPooling1D()(C1_a)


    #C1_out = Dropout(dropout)(C1_p)
    C1_out = C1_p

    #..................................

    C2_c= Conv1D(filters=cn2_kernel_num, kernel_size=cn2_kernel_size,
                 input_shape=input_sh,  kernel_regularizer=l2(beta),
                 padding='same')(input_C1_C2)
    C2_a = Activation(cn_activation)(C2_c)


    # il max pooling e' fatto usando tutto il vettore di ingresso
    # cosi' se e' presente la sequenza corrispondente ad un kernel
    # si accende l'outout
    # essendo k=3 abbiamo 64 configurazioni possibili, quindi
    # il numero di kernel puo' essere di poche decine
    # il numero di pesi da calcolare diventa molto piccolo
    # perche' ce' una uscita (che dovrebbe essere "binaria") per ogni kernel
    #sh = K.int_shape(C2_a)
    #C2_p = MaxPooling1D(pool_size=sh[1])(C2_a)

    C2_p = MaxPooling1D()(C2_a)

    #C2_out = Dropout(dropout)(C2_p)
    C2_out = C2_p

    sez_iniziale = [C1_out, C2_out]

    # out_C1_C2 = Merge(mode = "concat")(sez_iniziale)
    out_C1_C2 = Concatenate(axis = -1)(sez_iniziale)

    conv_model = Model(input = input_C1_C2, output=out_C1_C2)

    model = Sequential()

    model.add(conv_model)


    #..................
    model.add(Flatten())

    #.................
    # model.add(Dense(1, kernel_regularizer=l2(beta), activation='sigmoid'))
    model.add(Dense(dl1_units_num, kernel_regularizer=l2(beta), activation=dl1_activation))
    #..................
    model.add(Dropout(dropout))
    #..................
    model.add(Dense(dl2_units_num, kernel_regularizer=l2(beta), activation=dl2_activation))
    #..................

    optim = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss='binary_crossentropy')

    return model


def visualizza(model, nomeFile):
    plot_model(model, to_file=nomeFile)
