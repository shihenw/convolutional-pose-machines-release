# Convolutional Pose Machines
Shih-En Wei, Varun Ramakrishna, Takeo Kanade, Yaser Sheikh, "[Convolutional pose machines](http://arxiv.org/abs/1602.00134)", CVPR 2016.

Contact: shihenw@cmu.edu.

## Before Everything
- Watch some [videos](https://www.youtube.com/playlist?list=PLNh5A7HtLRcpsMfvyG0DED-Dr4zW5Lpcg).
- Install [Caffe](http://caffe.berkeleyvision.org/). If you are interested in training this model on your own machines, consider using [our version](https://github.com/shihenw/caffe) with a data layer performing online augmentation. Make sure you have done `make matcaffe` and `make pycaffe`.
- Copy `caffePath.cfg.example` to `caffePath.cfg` and set your own path in it.

## Testing
- Run `testing/get_model.sh` to retreive trained models from our web server.
- 1. `CPM_demo.m`: Put the testing image into `sample_image` then run it! You can select models (we provided 4) or other parameters in `config.m`. If you just want to try our best-scoring model, leave them default.
- 2. `CPM_benchmark.m`: Run the model on test benchmark and see the scores. Prediction files will be saved in `testing/predicts`.
- Python version (coming soon)

## Training
- Run `get_data.sh` to get datasets including [FLIC Dataset](http://vision.grasp.upenn.edu/cgi-bin/index.php?n=VideoLearning.FLIC), [LEEDS Sport Dataset](http://www.comp.leeds.ac.uk/mat4saj/lsp.html) and its [extended training set](http://www.comp.leeds.ac.uk/mat4saj/lspet.html), and [MPII Dataset](http://human-pose.mpi-inf.mpg.de/).
- Run `genJSON(<dataset_name>)` to generate a json file in `training/json/` folder. Dataset name can be `MPI`, `LEEDS`, or `FLIC`. The json files contain raw informations needed for training from each individual dataset.
- Run `python genLMDB.py` to generate LMDBs for CPM data layer in [our caffe](https://github.com/shihenw/caffe). Change the main function to select dataset, and note that you can generate a LMDB with multiple datasets.
- Run `python genProto.py` to get prototxt for caffe. Read [further explanation](https://github.com/shihenw/caffe) for layer parameters.
- Train with generated prototxts and collect caffemodels.

## Citation
Please cite CPM in your publications if it helps your research:

    @inproceedings{wei2016cpm,
        author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
        booktitle = {CVPR},
        title = {Convolutional pose machines},
        year = {2016}
    }
