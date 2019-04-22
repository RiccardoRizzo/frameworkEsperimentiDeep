#
# QUESTO FILE FA GIRARE TUTTO L'ESPERIMENTO
#


import local_lib
import run_kfold_2_0 as re

import sys
import os
# import h5py
sys.path.append("/home/riccardo/sw2/Stabili/lib/mail/")
import send_mail as sm
import datiEmail as dm

import time
import yaml


def main(filePar):
    """Lancia l'esperimento.

    In un file ricopia il file della rete ed il file dei parametri.
    Salva anche un grafico della rete.

    Salva le prestazioni medie della rete durante i fold ed invia una mail con i
    risultati a esperimento concluso.


    """
    # leggo il file della rete neurale come stringa
    with open(filePar, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
        pathNetwork = cfg["Alg"]["path"]

        nomeFileNetwork = cfg["Alg"]["file"]

        nDirOut = cfg["file"]["outputDir"]

    nomeFileRete=pathNetwork + nomeFileNetwork +".py"
    print(nomeFileRete)
    # leggo il file della rete neurale come stringa
    fin = open(nomeFileRete, "r")
    file_rete = fin.readlines()
    fin.close()

    # leggo il file dei parametri come stringa
    fin = open(filePar, "r")
    parametri = fin.readlines()
    fin.close()


    # Genera il nome del file di output
    nFOut = local_lib.nomeFileOut(filePar)

    # controlla che esista la dir, altrimenti la crea

    if not os.path.exists(nDirOut):
        os.makedirs(nDirOut)

    nFilePre = os.path.join(nDirOut, nFOut)

    # apre il file
    #>>>>>>>>>>>>>>>>>>>>>>
    with open(nFilePre + "_descr_rete.net", "w") as f:
        # inserisce le informazioni nel file dei risultati

        #=======================================================
        # scrive le stringhe del file della rete e dei parametri
        f.write("File_rete\n")
        f.write("".join(file_rete))

        f.write("Parametri_passati\n")
        f.write("".join(parametri))
        # =======================================================

    print("carica il dataset e crea la lista di indici del k-fold")
    df, fold = local_lib.prepara_ambiente(filePar)

    # esegue il test ottenendo le medie
    # dei parametri di output
    print("eseguo l'esperimento")

    ### tempo di run <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    start_time = time.time()

    M_acc, M_prec, M_rec = re.test(df, fold, filePar )

    ### TEMPO DI run <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    runtime = (time.time() - start_time)

    # scrivere i risultati in un file
    #nFOut = local_lib.nomeFileOut(filePar)

    # apre il file
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    with open(nFilePre + "_totale_training.csv", "w") as f:
        f.write("Tempo totale di run (in sec)" + ","+ str(runtime) + "\n")
        f.write("Media Accuracy"+ "," + str(M_acc)+ "\n")
        f.write("Media Precision"+","+ str(M_prec)+ "\n")
        f.write("Media Recall"+","+ str(M_rec)+ "\n")


    # email risultati
    subject = "Esperimenti prova " + nFilePre
    body = "finiti tutti gli esperimenti" + "\n"
    body +=  "Accuracy media :" + str( M_acc) + "\n"
    body +=  "Precision media:" + str( M_prec) + "\n"
    body +=  "Recall media:" + str( M_rec) + "\n"

    sm.send_email(dm.user, dm.pwd, dm.recipient, subject, body)


    # mandare la email

if __name__=="__main__":

    main(sys.argv[1])
