# Convolutional-Pose-Machines

## Before Everything
- Install [Caffe](http://caffe.berkeleyvision.org/). If you are interested train on this model, consider using [our version]() with an online augmentation layer. Make sure you have done `make matcaffe` and `make pycaffe`.
- Set Caffe path in `caffePath.cfg`.
- Run `get_data.sh` to get stuff to play with! 

## Testing
- Run `get_model.sh` to retreive trained models from our web server.
- 1. `CPM_Demo.m`: Put the testing image into `sample_image` then run it! You can select models (we provided 4) or other parameters in `config.m`.
- 2. `CPM_benchmark.m`: Run the model on test benchmark and see the scores.

## Training:
- Coming soon
