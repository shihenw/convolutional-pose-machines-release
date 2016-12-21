%% Demo code of "Convolutional Pose Machines", 
% Shih-En Wei, Varun Ramakrishna, Takeo Kanade, Yaser Sheikh
% In CVPR 2016
% Please contact Shih-En Wei at weisteady@gmail.com for any problems or questions

close all;
addpath('src'); 
addpath('util');
param = config();

%% choose a switch case
selection = 2;
switch(selection)
    case 1
        benchmark_data = 'MPI';
        benchmark_modelID = 1; % MPI model
        % MPI score 90.1% total PCKh@0.5
    case 2
        benchmark_data = 'LSP';
        benchmark_modelID = 1; % MPI+LSP model
        % LSP score 90.5% total PCK@0.2
    case 3
        benchmark_data = 'MPI';
        benchmark_modelID = 1; % MPI+LSP model
        % MPI score 88.5% @ PCKh0.5
    case 4
        benchmark_data = 'MPI';
        benchmark_modelID = 2; % MPI model
        % MPI score 87.9% @ PCKh0.5
    case 5
        benchmark_data = 'FLIC';
        benchmark_modelID = 4; % FLIC model
        % FLIC score 95.03% wrist PCK@0.2, 97.59% elbow PCK@0.2
end

makeFigure = 0; % switch to 1 to see result visually for LSP and FLIC

%% run
prediction_file = run_benchmark(param, benchmark_data, benchmark_modelID, makeFigure);
fprintf('prediction file saved at %s\n', prediction_file);
