# Large-Scale Multi-Domain Belief Tracking with Knowledge Sharing
The implementation of the model proposed in the Large-Scale Multi-Domain Belief Tracking with Knowledge Sharing [paper](https://arxiv.org/pdf/1807.06517) 

## Requirements
Python 3+ with pip

## Setup
To install the python modules required to train and run the model

```pip install -r requirements.txt```

## Preprocessing
To download and pre-process the multi-domain belief tracking dialogues and pre-trained word embeddings 

```python preprocess.py```

## Training
To train the model 

```python main.py train [--args=value]```

Some of these args include:
* `--num_hid`: the size of the hidden layers - default is 50 (the paper uses 100)
* `--bidir/--no-bidir`: bidirectional vs forward only - default is bidirectional (only valid for a RNNs)
* `--net_type`: the type of the feature encoders, `gru` for GRU, `cnn` for CNN, `lstm` for LSTM - default is `lstm` ([See the paper](https://arxiv.org/pdf/1807.06517))
* `--batch_size`: the batch size - default is 64
* `--dev`: device use to train the model (`cpu` or `gpu`) - default is `gpu`
* `--model_url`: path to save the model or to resume training - default is `models/model-1`
* `--graph_url`: path to save the tensorboard graph for evaluation metrics (cross-entropy, accuracy ..etc) - default is `graphs/graph-1`

For example to train the cnn variant to the model with 8 batch size on the GPU

```python main.py train --batch_size=8 --net_type=cnn --dev=gpu```

## Testing
To evaluate the model on the test dataset

```python main.py test [--args=value]```

This uses the same arguments as above. It generates a log file at *results/log-1.txt*, which includes the original dialogues with the true labels and the model predictions. 

Currently there is a **bug** that shuffles the model predictions across dialogues. So to go around it, **make sure to use a batch size of 1**, i.e. `--batch_size=1`. 
