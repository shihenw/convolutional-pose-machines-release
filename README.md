# Convolutional Pose Machines
Shih-En Wei, Varun Ramakrishna, Takeo Kanade, Yaser Sheikh, "[Convolutional Pose Machines](http://arxiv.org/abs/1602.00134)", CVPR 2016.

This project is licensed under the terms of the GPL v2 license. By using the software, you are agreeing to the terms of the [license agreement](https://github.com/shihenw/convolutional-pose-machines-release/blob/master/LICENSE).

Contact: Shih-En Wei (weisteady@gmail.com)

![Teaser?](https://github.com/shihenw/convolutional-pose-machines-release/blob/master/testing/python/figures/teaser-github.png)

## Recent Updates
- Synced our fork of caffe with most recent version (Dec. 2016) so that Pascal GPUs can work (tested with CUDA 8.0 and CUDNN 5).
- Including a VGG-pretrained model in matlab (and also python) code. This model was used in CVPR'16 demo. It scores 90.1% on MPI test set, and can be trained in much shorter time than previous models.
- We are working on [releasing code](https://github.com/ZheC/Multi-Person-Pose-Estimation/) of our [new work in multi-person pose estimation](https://arxiv.org/abs/1611.08050) demonstrated in ECCV'16 (best demo award!).

## Before Everything
- Watch some [videos](https://www.youtube.com/playlist?list=PLNh5A7HtLRcpsMfvyG0DED-Dr4zW5Lpcg).
- Install [Caffe](http://caffe.berkeleyvision.org/). If you are interested in training this model on your own machines, or realtime systems, please use [our version](https://github.com/shihenw/caffe) (a submodule in this repo) with customized layers. Make sure you have compiled python and matlab interface. This repository at least runs on Ubuntu 14.04, OpenCV 2.4.10, CUDA 8.0, and CUDNN 5. The following assumes you use `cmake` to compile caffe in `<repo path>/caffe/build`.
[//]: # (- Copy `caffePath.cfg.example` to `caffePath.cfg` and set your own path in it.)
- Include `<repo path>/caffe/build/install/lib` in environment variable `$LD_LIBRARY_PATH`.
- Include `<repo path>/caffe/build/install/python` in environment variable `$PYTHONPATH`.

## Testing
First, run `testing/get_model.sh` to retreive trained models from our web server.

### Python
- This [demo file](https://github.com/shihenw/convolutional-pose-machines-release/blob/master/testing/python/demo.ipynb) shows how to detect multiple people's poses as we demonstrated in CVPR'16. For real-time performance, please read it for further explanation.

### Matlab
- 1. `CPM_demo.m`: Put the testing image into `sample_image` then run it! You can select models (we provided 4) or other parameters in `config.m`. If you just want to try our best-scoring model, leave them default.
- 2. `CPM_benchmark.m`: Run the model on test benchmark and see the scores. Prediction files will be saved in `testing/predicts`.


## Training
- Run `get_data.sh` to get datasets including [FLIC Dataset](http://vision.grasp.upenn.edu/cgi-bin/index.php?n=VideoLearning.FLIC), [LEEDS Sport Dataset](http://www.comp.leeds.ac.uk/mat4saj/lsp.html) and its [extended training set](http://www.comp.leeds.ac.uk/mat4saj/lspet.html), and [MPII Dataset](http://human-pose.mpi-inf.mpg.de/).
- Run `genJSON(<dataset_name>)` to generate a json file in `training/json/` folder (you'll have to create it). Dataset name can be `MPI`, `LEEDS`, or `FLIC`. The json files contain raw informations needed for training from each individual dataset.
- Run `python genLMDB.py` to generate LMDBs for CPM data layer in [our caffe](https://github.com/shihenw/caffe). Change the main function to select dataset, and note that you can generate a LMDB with multiple datasets.
- Run `python genProto.py` to get prototxt for caffe. Read [further explanation](https://github.com/shihenw/caffe) for layer parameters.
- Train with generated prototxts and collect caffemodels.

## Related Repository
- [Convolutional Pose Machines in Tensorflow](https://github.com/psycharo/cpm)

## Citation
Please cite CPM in your publications if it helps your research:

    @inproceedings{wei2016cpm,
        author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
        booktitle = {CVPR},
        title = {Convolutional pose machines},
        year = {2016}
    }
