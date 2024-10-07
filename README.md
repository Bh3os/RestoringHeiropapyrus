# HieroLM
This is the PyTorch implementation for HieroLM proposed in the paper **HieroLM: Egyptian Hieroglyph Recovery with Next Word
Prediction Language Model** submitted to NAACL 2025.


#### To train the model, run the following command:

```
python main.py --cuda True --dataset [aes/ramses/mixed]
```

#### To test the trained model on test set, run the following command:

```
python main.py --cuda True --dataset [aes/ramses/mixed] --mode decode
```

#### To interact with the trained model in real time, run the following command:

```
python main.py --cuda True --dataset [aes/ramses/mixed] --mode realtime
```
