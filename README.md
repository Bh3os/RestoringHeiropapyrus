# HieroLM
This is the PyTorch implementation for HieroLM proposed in the paper **HieroLM: Egyptian Hieroglyph Recovery with Next Word
Prediction Language Model** published at *The 9th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature (LaTeCH-CLfL 2025)* at *NAACL 2025*.

## Project Structure

```
HieroLM/
├── data/                       # Data directory
│   ├── aes/                    # AES dataset
│   ├── mixed/                  # Mixed dataset
│   └── ramses/                 # Ramses dataset
├── hierolm/                    # Main package
│   ├── __init__.py             # Package initialization
│   ├── model.py                # HieroLM model implementation
│   ├── vocab.py                # Vocabulary handling
│   ├── utils.py                # Utility functions
│   ├── parse.py                # Command line argument parsing
│   ├── train.py                # Training functions
│   ├── evaluation.py           # Evaluation functions
│   └── decode.py               # Decoding and inference functions
├── saved_models/               # Directory for saved models
├── requirements.txt            # Project dependencies
├── run.py                      # New main script to run the model
├── main.py                     # Legacy main script
└── README.md                   # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/HieroLM.git
cd HieroLM
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

You can use either the new `run.py` script (recommended) or the legacy `main.py` script.

### Training

To train the model:

```bash
python run.py --mode train --dataset aes --cuda
```

### Evaluation

To evaluate the model on test data:

```bash
python run.py --mode decode --dataset aes --model_path model.bin --cuda
```

### Multi-shot Evaluation

To evaluate multi-shot accuracy:

```bash
python run.py --mode multishot --dataset aes --model_path model.bin --cuda
```

### Interactive Mode

To use the model interactively:

```bash
python run.py --mode realtime --dataset aes --model_path model.bin --cuda
```

## Legacy Commands

If you prefer to use the original scripts:

#### To train the model:

```
python main.py --cuda True --dataset [aes/ramses/mixed]
```

#### To test the trained model on test set:

```
python main.py --cuda True --dataset [aes/ramses/mixed] --mode decode
```

#### To interact with the trained model in real time:

```
python main.py --cuda True --dataset [aes/ramses/mixed] --mode realtime
```
