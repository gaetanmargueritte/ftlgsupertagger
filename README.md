
# ftlgsupertagger

  

This is the code to support the paper "Multi-purpose neural network for French categorial grammars", Gaëtan Margueritte, Daisuke Bekki, Koji Mineshima, submitted to IWCS2023 (to appear in IWCS proceedings in June).

  

#### *Please note that the dataset provided is extracted from the TLGBank by Richard Moot (https://www.labri.fr/perso/moot/TLGbank/) in order to allow the results to be reproducible. It is licensed under GNU Lesser General Public License and downloading any of these files constitutes an agreement to all the terms of this license.*

  

## Installation

In order to run the project, we recommend you to use a virtual environment.

```

python3 -m venv py3

source py3/bin/activate

```

You can download dependencies automatically using `poetry`.

```

poetry install --no-root

```

It is recommended to install the correct version of `cuda` for your working station.

  
  

## Running the program

In order to train the neural model, please run the following command:

```

python3 train.py

```

You can also change the default parameters available using the command line, such as the batch size or the seed. For more information on arguments, please use the `-h` option.

```

python3 train.py -h

```

Files used during training:

-  `train.py` organizes the training phase of the model ;

-  `model.py` defines the architecture of the model ;

-  `tlgbank.txt` input file containing the training data ;

-  `dataHandle.py` contains methods used to loading data into Python structures ;

-  `utils.py` various utilities such as padding.

  

## Results and outputs

  

Running the program will create the following files:

-  `evaluation/temp/unrecognized_tags.txt`, the list of expected tags that were not recognized during evaluation. Used to identify that most tags are really complex (thus rare) ;

- several model files used to restart training from a selected phase (given as argument )

-- `model.pt` phase 1 trained model (`python3 train.py --model model.pt`) ;

-- `vae_model.pt` phase 2 trained model (`python3 train.py --vae-model vae_model.pt`) ;

-- `final_model.pt` phrase 3 trained model (no command, as no training is required) ;

-  `model_data.pickle` data used for model initialization, used to manually test sentences using `main.py`

  

Prints such as training statistics are indicated in `stdout` terminal. Loss and accuracy per tag class (e.g. Uncommon, Rare) for training/evaluation/test are presented.

  

## Tagging custom sentences

You can test custom sentences formulated in French using `main.py` and a trained model. To do so, please use the following command:

```

python3 main.py --model [trained model file] --data [model data pickle file] --input [input sentences in .txt format]

```

  

Examples are provided in the file `input.txt`.