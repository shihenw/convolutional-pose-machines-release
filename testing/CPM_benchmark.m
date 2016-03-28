close all;
addpath('src'); 
addpath('util');
param = config();

%% edit this part (select one to uncomment)
selection = 1;
switch(selection)
    case 1
        benchmark_data = 'LSP';
        benchmark_modelID = 1; % MPI+LSP model
        % LSP score 90.5% total PCK@0.2
    case 2
        benchmark_data = 'MPI';
        benchmark_modelID = 1; % MPI+LSP model
        % MPI score 88.5% @ PCKh0.5
    case 3
        benchmark_data = 'MPI';
        benchmark_modelID = 2; % MPI model
        % MPI score 87.9% @ PCKh0.5
    case 4
        benchmark_data = 'LSP';
        benchmark_modelID = 3; % LSP model 
        % LSP score 84.32% total PCK@0.2
    case 5
        benchmark_data = 'FLIC';
        benchmark_modelID = 4; % FLIC model
        % FLIC score 95.03% wrist PCK@0.2, 97.59% elbow PCK@0.2
end

makeFigure = 0; % switch to 1 to see result visually for LSP and FLIC

%% run
prediction_file = run_benchmark(param, benchmark_data, benchmark_modelID, makeFigure);
fprintf('prediction file saved at %s\n', prediction_file);
