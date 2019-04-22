# frameworkEsperimentiDeep


# Sistema per lanciare semplici esperimenti k-fold

Il sistema si aspetta un dataframe in cui sono presenti i campioni sulle righe e nelle colonne le rappresentazioni intese come vettori di features, una rappresentazione, e quindi un vettore, per ogni colonna. 

Il datagrame e' quindi fatto in questo modo:

![immagine del dataframe](/home/riccardo/sw2/Stabili/lib/RetiNeurali/frmwrkSimpleExp/Training/html/dataframe.png  "dataframe")

La creazione del dataframe si puo' vedere nel notebook [preprocessing dei dati](/home/riccardo/sw2/Stabili/lib/RetiNeurali/frmwrkSimpleExp/Training/html/Preprocessing dati.html) . Un altro esempio e' dato dal preprocessine dei dati di un sottoinsieme di MNIST in [questo file](/home/riccardo/sw2/Stabili/lib/RetiNeurali/frmwrkSimpleExp/Training/html/Pre processing dati MNIST semplificato.html)

Una volta creato il dataframe si passa alla scrittura del file di configurazione.

Il file di configurazione e' test-par.yaml ed e' spiegato sotto:

	#
	# IL FILE RACCOGLIE TUTTI I PARAMETRI DELL'ESPERIMENTO
	#
	# Si suppone che i dati in ingresso siano in un file in cui si trova
	# un sample per ogni riga. La prima riga contiene le intestazioni delle
	# colonne.
	#
	# La procedura e' pensata per esperimenti piccoli, in quanto tutto e' fatto
	# in memoria.
	#
	# Dopo il caricamento si suppone di avere un dataframe cosi' fatto
	#
	#               features
	#            -----------------
	#           |                 |
	#           |                 |
	# campioni  |                 |
	#           |                 |
	#           |                 |
	#           |                 |
	#           ------------------
	#
	# Delle features una o piu' sono quelle di uscita:
	#   l_input contiene le intestazioni delle colonne di ingresso
	#   l_output contiene le intestazioni delle colonne di uscita
	#


La parte dei parametri dell'esperimento: 

	parEsp :
	    k : 10 # 10 se si vuole eseguire il 10 fold

	    l_input : ["immagine"]   # lista delle colonne di input

	    l_output : ["classe"] # lista delle colonne di uscita
	    l_classi : ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

Nome dei file contenenti la rete


	Alg :
	  path : "./" # path del file della rete neurale
	  file : "conv2_2mlp"  # nome del file della rete neurale


Parametri dela rete

	parAlg :

	  input_shape : [784, 1]

	  dropout : 0.5
	  beta : 1e-3


	  cn1_kernel_num : 30
	  cn1_kernel_size : 20

	  cn2_kernel_num : 50
	  cn2_kernel_size : 3

	  cn_activation  :  "relu"

	  dl1_units_num : 300
	  dl1_activation : "relu"


	  dl2_units_num : 10
	  dl2_activation : "softmax"


	  validation_split : 0.1
	  learning_rate : 0.00005
	  batch_size : 64
	  epochs : 2

	  norma : False

Nome dei file coinvolti 



	  inputFile : "/home/riccardo/sw2/Stabili/lib/RetiNeurali/frmwrkSimpleExp/prova/test_dataframe.pkl" # path completo del file di input
	  outputDir : "./DirOut/" # directory di output
	  outputFile : "Prova_MNIST" # prefisso del file di output
	  
	  
Il file di parametri si deve dare come input al file main con una stringa del tipo 

	python main_esp_2_0.py test_par.yaml 
	
Documentazione relativa ai file [main_esp_2_0.py](./main_esp_2_0.html)
Documentazione relativa ai file [run_kfold_2_0.py](/home/riccardo/sw2/Stabili/lib/RetiNeurali/frmwrkSimpleExp/Training/html/run_kfold_2_0.html)
Documentazione relativa ai file [local_lib.py](/home/riccardo/sw2/Stabili/lib/RetiNeurali/frmwrkSimpleExp/Training/html/local_lib.html)
