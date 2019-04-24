



import local_lib as ll
# import vis_modello as vs

import numpy as np
import sklearn

import yaml

import matplotlib.pyplot as plt

# import h5py

from importlib import import_module

from keras.utils import plot_model 

import time
import keras

import os



#---------------------------------------------------------
def dynamic_import(abs_module_path, modulo):
    """Import dinamico del modulo

    Args:
        abs_module_path: path assoluto del modulo
        modulo: stringa del nome del modulo da importare

    Returns:
        nome del modulo da importare
    """
    module_object = import_module(abs_module_path)

    out = getattr(module_object, modulo)

    return out


#...............................................................................
def test(df, fold, filePar):
    """Fa girare tutti i fold

    La funzione esegue tutti i fold usando i dati nel dataframe df
    e consegna i risultati di accuracy, precsion e recall medi.

    Il modello della rete e' letto dal file indicato nel file parametri
    e viene reinizializzato ogni volta. La rete che raggiunge il miglior
    risultato e' memorizzata in un file .h5

    Il parametri fold deve contenere la lista delle liste delle righe usate
    per ogni fold. I risultati ottenuti per ogni fold sono memorizzati
    in un file *_risultati.yaml

    Args:
        df: dataframe da fornire per il test
        fold: lista di liste di righe del dataframe
            (le singole liste costituiscono un fold)
    Returns:
        M_acc: accuracy media nei vari fold
        M_prec: precision media nei vari fold
        M_rec: recall media nei vari fold
    """

    acc_t=[]
    prec_t=[]
    rec_t=[]

    with open(filePar, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        # carico le classi
        l_classi = cfg["parEsp"]["l_classi"]

        nDirOut = cfg["file"]["outputDir"]

    acc_BEST = 0.0
    fold_BEST = -1
    for i in range(len(fold)):

        # esegue il training della rete
        y_pred , Y_test, y_out, runtime, rete, storia = run_fold(df, fold, filePar, i)


        # controlla le prestazioni della rete nel fold i
        acc_fold = sklearn.metrics.accuracy_score(Y_test, y_pred, normalize=True,  sample_weight=None)
        prec_fold = sklearn.metrics.precision_score(Y_test, y_pred, average="micro")
        rec_fold = sklearn.metrics.recall_score(Y_test, y_pred, average="micro")
        confusion_fold = sklearn.metrics.confusion_matrix(Y_test, y_pred, labels = l_classi)

        # se le prestazioni sono migliori di quelle precedenti salva il modello
        # in memoria
        if acc_fold > acc_BEST :
            acc_BEST = acc_fold
            rete_BEST = rete
            fold_BEST = i


        # salva le prestazioni nella lista
        acc_t.append(acc_fold)
        prec_t.append(prec_fold)
        rec_t.append(rec_fold)

        # Scrive i primi dati dell'esperimento in un file
        nFOut = ll.nomeFileOut(filePar)
        nFilePre = os.path.join(nDirOut, nFOut)

        sep = " :\n"
        with open(nFilePre + "_risultati.yaml", "a") as f:
            f.write("fold"+sep + str(i) + "\n")
            f.write("lista_pat"+sep + np.array2string(fold[i]["test"]) +"\n")
            f.write("predizioni"+sep + np.array2string(np.array(y_pred)) +"\n")
            f.write("valore_vero"+sep + np.array2string(np.array(Y_test)) +"\n")
            f.write("vettore_output"+sep + np.array2string(np.array(y_out)) +"\n")
            f.write("runtime"+sep + str(runtime) +"\n")
            #
            f.write("acc_fold"+sep + str(acc_fold) +"\n")
            f.write("prec_fold"+sep + str(prec_fold) +"\n")
            f.write("rec_fold"+sep + str(rec_fold) +"\n")
            f.write("confusion"+sep + np.array2string(confusion_fold) +"\n")
            f.write("#\n#\n")

        # memorizza una immagine con la storia dell'apprendimento
        plt.plot(storia.history['acc'])
        plt.plot(storia.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch_' + str(i))
        plt.legend(['Train', 'Test'], loc='upper left')
        nFile = nFilePre + "_fold_" + str(i) + ".png"
        plt.savefig(nFile)
        plt.close()

    # calcola la media delle prestazioni
    M_acc=np.average(acc_t)
    M_prec=np.average(prec_t)
    M_rec=np.average(rec_t)

    nFOut = ll.nomeFileOut(filePar) + "_network_fold_" + str(fold_BEST)+ ".h5"
    nFileNet = os.path.join(nDirOut, nFOut)
    rete_BEST.save(nFileNet)

    nFOut = ll.nomeFileOut(filePar) + "_arch_net" + str(fold_BEST)
    nFileNet = os.path.join(nDirOut, nFOut)
    #vs.main(nFileNet, rete_BEST)

    # scrive l'immagine della struttura della rete
    nFOut = ll.nomeFileOut(filePar) + "_immagine_" + str(fold_BEST) + ".png"
    nFileNet = os.path.join(nDirOut, nFOut)
    plot_model(rete_BEST, to_file=nFileNet, show_shapes=True, show_layer_names=True)


    return M_acc, M_prec, M_rec



#...............................................................................
def run_fold(df, fold, filePar, i):
    """Esegue il run del fold i dell'esperimento.

    Dal filePar si leggono i parametri della rete neurale.
    Dal fold sono separati i dati di training ed i dati di test, si esegue il training
    e si esegue il test, Le previsioni sono portate in output. Il calcolo
    dei risultati dell'apprendimento e' fatto nel programma chiamante.

    Args:
        df: dataframe contenente tutti i dati
        fold: lista di liste di righe del dataframe (le singole liste costituiscono un fold)
        filePar: nome del file di parametri della rete neurale
        i: numero del fold da fare girare

    Returns:
        y_pred: lista di valori corrispondenti alle risposte della rete
        Y_test: lista di valor di riferimento, contenuti del dataframe df
        y_out: array corrispondente ai valori delle uscite della
                rete per ogni pattern di ingresso
        runtime: tempo in secondi impiegato per il training della rete

    """

    import sys


    # legge dal file di parametri il nome del file con la rete
    with open(filePar, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        # carico le classi
        l_classi = cfg["parEsp"]["l_classi"]

        # # CARICO I PARAMETRI DELLA RETE
        input_shape = cfg["parAlg"]["input_shape"]

        num_output = cfg["parAlg"]["output_shape"]


        validation_split = cfg["parAlg"]["validation_split"]
        # learning_rate = cfg["parAlg"]["learning_rate"]
        batch_size = cfg["parAlg"]["batch_size"]
        epochs = cfg["parAlg"]["epochs"]
        norma = cfg["parAlg"]["norma"]

        pathNetwork =  cfg["Alg"]["path"]
        sys.path.append( pathNetwork)

        nomeFileNetwork = cfg["Alg"]["file"]

    # importa la libreria delle reti neurali
    riga = "import " + nomeFileNetwork + " as NN"
    exec(riga)  # importa il file con la rete

    # faccio l'import dinamico della funzione che contiene la rete neurale
    rete_neurale = dynamic_import(nomeFileNetwork, "Net_f")


    l_features, l_output, _, _, _ =ll.leggeParametri(filePar)

    # seleziona i dati di ingresso e di uscita
    X_train, Y_train = ll.estraiDati(fold[i]["train"], df, l_features, l_output, norma)
    X_test, Y_test = ll.estraiDati(fold[i]["test"], df, l_features, l_output, norma)


    # model = NN.Net_f(filePar)
    model = rete_neurale(filePar)

    ### TEMPO DI ADDESTRAMENTO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    start_time = time.time()


    # esegue il training
    X=np.reshape(X_train, (len(X_train) , input_shape[0], input_shape[1] ) )

    # trasforma Y da [XXX, 1] a [XXX,] ????
    Y = np.reshape(Y_train, (len(Y_train) ,) )
    # trasforma il numero della classe (classeID) in categoria
    Y = keras.utils.to_categorical(Y, num_output)


    history = model.fit(X, Y, validation_split=validation_split, batch_size=batch_size, epochs=epochs, verbose =0)
    # "TEMPO DI ADDESTRAMENTO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    runtime = (time.time() - start_time)

    # PREDIZIONI =============================================================
    # genera l'output del modello
    X = np.reshape(X_test, (len(X_test), input_shape[0], input_shape[1] ) )
    y_out = model.predict(X)

    y_pred = [ np.argmax(x) for x in y_out ]
    y_pred = [l_classi[i] for i in y_pred]
    # confronto con la soglia
    #y_pred = y_out > dl2_soglia
    #y_pred.astype(np.int)
    # =======================================================================

    # trasforma Y da [XXX, 1] a [XXX,] ????
    Y_test = list( np.reshape(Y_test, [len(Y_test), ]) )
    Y_test = [l_classi[int(i)] for i in Y_test]
    # trasforma il numero della classe (classeID) in categoria
    #Y_test = keras.utils.to_categorical(Y_test, dl2_units_num)

    #y_pred = np.reshape(y_pred, [len(y_pred), ])

    #print "y_pred", y_pred
    #print "Y_test", Y_test
    #print "y_out", y_out

    return y_pred, Y_test, y_out, runtime, model, history
