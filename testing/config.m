function param = config()
%% set this part

% CPU mode (0) or GPU mode (1)
% friendly warning: CPU mode may take a while
param.use_gpu = 1;

% GPU device number (doesn't matter for CPU mode)
GPUdeviceNumber = 0;

% Select model (default: 1)
% 1: MPII+LSP(PC) 6-stage CPM
% 2: MPII 6-stage CPM
% 3: LSP(PC) 6-stage CPM
% 4: FLIC 4-stage CPM (upper body only)
param.modelID = 1;

% Scaling paramter: starting and ending ratio of person height to image
% height, and number of scales per octave
% warning: setting too small starting value on non-click mode will take
% large memory
param.octave = 6;


%% don't edit this part

% path of your caffe
caffepath = textread('../caffePath.cfg', '%s', 'whitespace', '\n\t\b ');
caffepath= [caffepath{1} '/matlab/'];
fprintf('You set your caffe in caffePath.cfg at: %s\n', caffepath);
addpath(caffepath);
caffe.reset_all();
if(param.use_gpu)
    fprintf('Setting to GPU mode, using device ID %d\n', GPUdeviceNumber);
    caffe.set_mode_gpu();
    caffe.set_device(GPUdeviceNumber);
else
    fprintf('Setting to CPU mode.\n');
    caffe.set_mode_cpu();
end

param.click = 1;

param.model(1).caffemodel = '../model/_trained_MPI/pose_iter_985000_addLEEDS.caffemodel';
param.model(1).deployFile = '../model/_trained_MPI/pose_deploy_centerMap.prototxt';
param.model(1).description = 'MPII+LSP 6-stage CPM';
param.model(1).description_short = 'MPII_LSP_6s';
param.model(1).boxsize = 368;
param.model(1).padValue = 128;
param.model(1).np = 14;
param.model(1).sigma = 21;
param.model(1).stage = 6;
param.model(1).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(1).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
                     
param.model(2).caffemodel = '../model/_trained_MPI/pose_iter_630000.caffemodel';
param.model(2).deployFile = '../model/_trained_MPI/pose_deploy_centerMap.prototxt';
param.model(2).description = 'MPII 6-stage CPM';
param.model(2).description_short = 'MPII_6s';
param.model(2).boxsize = 368;
param.model(2).padValue = 128;
param.model(2).np = 14;
param.model(2).sigma = 21;
param.model(2).stage = 6;
param.model(2).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(2).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
                     
param.model(3).caffemodel = '../model/_trained_LEEDS_PC/pose_iter_395000.caffemodel';
param.model(3).deployFile = '../model/_trained_LEEDS_PC/pose_deploy_centerMap.prototxt';
param.model(3).description = 'LSP (PC) 6-stage CPM';
param.model(3).description_short = 'LSP_6s';
param.model(3).boxsize = 368;
param.model(3).np = 14;
param.model(3).sigma = 21;
param.model(3).stage = 6;
param.model(3).padValue = 128;
param.model(3).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(3).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
                     
param.model(4).caffemodel = '../model/_trained_FLIC/pose_iter_40000.caffemodel';
param.model(4).deployFile = '../model/_trained_FLIC/pose_deploy.prototxt';
param.model(4).description = 'FLIC (upper body only) 4-stage CPM';
param.model(4).description_short = 'FLIC_4s';
param.model(4).boxsize = 368;
param.model(4).np = 9;
param.model(4).sigma = 21;
param.model(4).stage = 4;
param.model(4).padValue = 128;
param.model(4).limbs = [1 2; 2 3; 4 5; 5 6];
param.model(4).part_str = {'Lsho', 'Lelb', 'Lwri', ...
                           'Rsho', 'Relb', 'Rwri', ...
                           'Lhip', 'Rhip', 'head', 'bkg'};
