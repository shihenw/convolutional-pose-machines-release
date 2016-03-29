%% Demo code of "Convolutional Pose Machines", 
% Shih-En Wei, Varun Ramakrishna, Takeo Kanade, Yaser Sheikh
% In CVPR 2016
% Please contact Shih-En Wei at shihenw@cmu.edu for any problems or questions
%%
close all;
addpath('src'); 
addpath('util');
addpath('util/ojwoodford-export_fig-5735e6d/');
param = config();

fprintf('Description of selected model: %s \n', param.model(param.modelID).description);

%% Edit this part
% put your own test image here
%test_image = 'sample_image/singer.jpg';
%test_image = 'sample_image/shihen.png';
test_image = 'sample_image/roger.png';
%test_image = 'sample_image/nadal.png';
%test_image = 'sample_image/LSP_test/im1640.jpg';
%test_image = 'sample_image/CMU_panoptic/00000998_01_01.png';
%test_image = 'sample_image/CMU_panoptic/00004780_01_01.png';
%test_image = 'sample_image/FLIC_test/princess-diaries-2-00152201.jpg';
interestPart = 'Lwri'; % to look across stages. check available names in config.m

%% core: apply model on the image, to get heat maps and prediction coordinates
figure(1); 
imshow(test_image);
hold on;
title('Drag a bounding box');
rectangle = getrect(1);
[heatMaps, prediction] = applyModel(test_image, param, rectangle);

%% visualize, or extract variable heatMaps & prediction for your use
visualize(test_image, heatMaps, prediction, param, rectangle, interestPart);