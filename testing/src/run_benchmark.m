function prediction_file = run_benchmark(param, benchmark_data, benchmark_modelID, makeFigure)

model = param.model(benchmark_modelID);
scale_search = 0.7:0.1:1.3; % fit training
boxsize = model.boxsize;
np = model.np;
obj = zeros(1,np);
detected = zeros(1,np);

% some dataset specific metadata
if(strcmp(benchmark_data, 'MPI'))
    fprintf('load MPI mat file....\n');
    gt = load('../dataset/MPI/mpii_human_pose_v1_u12_1/mpii_human_pose_v1_u12_1.mat');
    RELEASE_img_index = find(~gt.RELEASE.img_train);
    order_to_standard = [8 7 11 10 9 12 13 14 3 2 1 4 5 6];
    RELEASE_predicted = gt.RELEASE;
    count = 0;
    order_to_MPI = [0 1 2 3 4 5 8 9 10 11 12 13 14 15];
    target_dist = 41/35; % fit training
elseif(strcmp(benchmark_data, 'LSP'))
    fprintf('load LSP mat file....\n');
    gt = load('../dataset/LEEDS/lsp_dataset/joints.mat');
    testLength = 1000;
    order_to_lsp = [14 13 9 8 7 10 11 12 3 2 1 4 5 6];
    target_dist = 0.8; % the only clue is image height
elseif(strcmp(benchmark_data, 'FLIC'))
    fprintf('load FLIC mat file....\n');
    gt = load('../dataset/FLIC/examples.mat');
    testIdx = find([gt.examples.istest]);
    testLength = length(testIdx);
    scale_constant = 0.0110;
    target_dist = 41/35; % fit training
else
    error('wrong dataset name');
end

net = caffe.Net(model.deployFile, model.caffemodel, 'test');
center_map = produceCenterLabelMap([boxsize boxsize], boxsize/2, boxsize/2, model.sigma);

fprintf('Running inference on %s using model %s, %d scales for each sample.\n', benchmark_data, model.description, length(scale_search));

if(strcmp(benchmark_data, 'MPI'))
    for i = 1:length(RELEASE_img_index)
        img_index = RELEASE_img_index(i);

        for person_index = 1:length(gt.RELEASE.annolist(img_index).annorect)
            count = count + 1;
            fprintf('Testing MPI: image %d/%d, person %d, count: %d\n', i, length(RELEASE_img_index), person_index, count);

            imagePath = gt.RELEASE.annolist(img_index).image.name;
            imagePath = ['../dataset/MPI/images/' imagePath];

            % load image, objpos, and scale, but might fail
            try
                oriImg = imread(imagePath);
            catch
                error('image cannot be loaded, make sure you have %s', imagePath);
            end

            try
                objpos = gt.RELEASE.annolist(img_index).annorect(person_index).objpos;
                center = [objpos.x objpos.y];
                scale_provided = gt.RELEASE.annolist(img_index).annorect(person_index).scale;
                scale0 = target_dist/scale_provided;
            catch
                fprintf('!!!!!!!!!!!!!!!!!!skipped at img_index %d and person_index %d\n', i, person_index);
                count = count - 1;
                continue;
            end

            multiplier = scale_search;
            score = cell(1, length(multiplier));
            pad = cell(1, length(multiplier));

            for m = 1:length(multiplier)
                scale = scale0 * multiplier(m);
                imageToTest = imresize(oriImg, scale);

                center_s = center * scale;
                [imageToTest, pad{m}] = padAround(imageToTest, boxsize, center_s);
                imageToTest = preprocess(imageToTest, 0.5, center_map);

                score{m} = applyDNN(imageToTest, net);
                pool_time = size(imageToTest,1) / size(score{m},1);
                % post-processing the heatmap
                score{m} = imresize(score{m}, pool_time);
                score{m} = resizeIntoScaledImg(score{m}, pad{m});
                score{m} = imresize(score{m}, [size(oriImg,2) size(oriImg,1)]);
            end

            % summing up scores
            final_score = zeros(size(score{1}));
            for m = 1:size(score,2)
                final_score = final_score + score{m};
            end
            final_score = permute(final_score, [2 1 3]); 

            % ----- generate prediction -----
            prediction = zeros(np,2);
            for j = 1:np
                [prediction(j,2), prediction(j,1)] = findMaximum(final_score(:,:,j));
            end
            prediction(order_to_standard,:) = prediction;

            for j = 1:np
                RELEASE_predicted.annolist(img_index).annorect(person_index).annopoints.point(j).id = order_to_MPI(j);
                RELEASE_predicted.annolist(img_index).annorect(person_index).annopoints.point(j).x = prediction(j,1);
                RELEASE_predicted.annolist(img_index).annorect(person_index).annopoints.point(j).y = prediction(j,2);
            end
        end
    end
    prediction_file = sprintf('predicts/MPI_prediction_model_%s.mat', model.description_short);
    save(prediction_file, 'RELEASE_predicted');

elseif(strcmp(benchmark_data, 'LSP'))
    joint_gt = zeros(np, 2, testLength);
    prediction_all = zeros(np, 2, testLength);
    
    for i = 1:testLength
        fprintf('image %d/%d', i, testLength);
        imagePath = sprintf('../dataset/LEEDS/lsp_dataset/images/im%04d.jpg', i+1000);
        try
            oriImg = imread(imagePath);
        catch
            error('image cannot be loaded, make sure you have %s', imagePath);
        end

        center = [size(oriImg,2), size(oriImg,1)]/2;
        scale_provided = size(oriImg,1)/model.boxsize; % something prop to image height
        scale0 = target_dist/scale_provided;

        joint_gt(:,:,i) = gt.joints(1:2,:,1000+i)';

        multiplier = scale_search;
        score = cell(1, length(multiplier));
        pad = cell(1, length(multiplier));

        for m = 1:length(multiplier)
            scale = scale0 * multiplier(m);
            imageToTest = imresize(oriImg, scale);

            center_s = center * scale;
            [imageToTest, pad{m}] = padAround(imageToTest, boxsize, center_s);

            imageToTest = preprocess(imageToTest, 0.5, center_map);

            score{m} = applyDNN(imageToTest, net);
            pool_time = size(imageToTest,1) / size(score{m},1);
            score{m} = imresize(score{m}, pool_time);
            score{m} = resizeIntoScaledImg(score{m}, pad{m});
            score{m} = imresize(score{m}, [size(oriImg,2), size(oriImg,1)]);        
        end

        % summing up scores
        final_score = zeros(size(score{1,1}));
        for m = 1:size(score,2)
            final_score = final_score + score{m};
        end
        final_score = permute(final_score, [2 1 3]); 
        % generate prediction
        prediction = zeros(np,2);
        for j = 1:np
            [prediction(j,2), prediction(j,1)] = findMaximum(final_score(:,:,j));
        end
        prediction(order_to_lsp,:) = prediction;
        final_score(:,:,order_to_lsp) = final_score(:,:,1:np);

        for j = 1:np
            if(makeFigure)
                max_value = max(max(final_score(:,:,j)));
                imToShow = single(oriImg)/255 * 0.5 + mat2im(final_score(:,:,j), jet(100), [0 max_value])/2;
                imToShow = insertShape(imToShow, 'FilledCircle', [prediction(j,:) 2], 'Color', 'w'); 
                imToShow = insertShape(imToShow, 'FilledCircle', [joint_gt(j,1:2,i) 2], 'Color', 'g'); 
                imToShow = insertShape(imToShow, 'FilledRectangle', [center 3 3], 'Color', 'c');
                
                figure(1); imshow(imToShow);
                title('paused, click to resume');
                pause;
            end

            bodysize = util_get_bodysize_size(joint_gt(:,:,i));
            error_dist = norm(prediction(j,:) - joint_gt(j,1:2,i));
            hit = error_dist <= bodysize*0.2;
            
            obj(j) = obj(j) + 1;
            if(hit)
                detected(j) = detected(j) + 1;
            end
            fprintf(' %d', hit);
        end

        for j = 1:np
            fprintf(' %.3f', detected(j)/obj(j));
        end
        fprintf(' |%.4f\n', sum(detected)/sum(obj));

        prediction_all(:,:,i) = prediction;
    end
    prediction_file = sprintf('predicts/LSP_prediction_model_%s.mat', model.description_short);
    save(prediction_file, 'prediction_all');

elseif(strcmp(benchmark_data, 'FLIC'))
    prediction_all = zeros(np, 2, testLength);
    
    for i = 1:testLength
        idx = testIdx(i);

        fprintf('%d/%d: %d', i, testLength, idx);
        imagePath = gt.examples(idx).filepath;
        imagePath = ['../dataset/FLIC/images/' imagePath];

        gt_joint = gt.examples(idx).coords;
        gt_joint(isnan(gt_joint)) = [];
        gt_joint = reshape(gt_joint, [2 11]);
        gt_joint(:,[9 10]) = [];
        gt_joint = gt_joint';
        gt_joint = [gt_joint, ones(9,1)];

        if(gt_joint(4,1) > gt_joint(1,1)) % Observer-centric annotation
            gt_joint([1 2 3 4 5 6 7 8],:) = gt_joint([4 5 6 1 2 3 8 7],:);
        end

        box = gt.examples(idx).torsobox;
        torso_height = box(4) - box(2);
        center = [(box(1)+box(3))/2, (box(2)*0.7 + box(4)*0.3)];

        oriImg = imread(imagePath);
        scale_provided = scale_constant * torso_height;
        scale0 = target_dist/scale_provided;

        multiplier = scale_search;
        score = cell(1, length(multiplier));
        peakValue = zeros(length(multiplier), np+1);
        pad = cell(1, length(multiplier));
        
        for m = 1:length(multiplier)
            scale = scale0 * multiplier(m);
            imageToTest = imresize(oriImg, scale);

            center_s = center * scale;
            [imageToTest, pad{m}] = padAround(imageToTest, boxsize, center_s);

            imageToTest = preprocess(imageToTest, 0.5, center_map);
            score{m} = applyDNN(imageToTest, net);
            pool_time = size(imageToTest,1) / size(score{m},1);
            score{m} = imresize(score{m}, pool_time);
            score{m} = resizeIntoScaledImg(score{m}, pad{m});
            score{m} = imresize(score{m}, [size(oriImg,2) size(oriImg,1)]);
        end
        
        % summing up scores
        final_score = zeros(size(score{1}));
        for m = 1:size(score,2)
            final_score = final_score + score{m};
        end
        final_score = permute(final_score, [2 1 3]);
        % ----- generate prediction -----
        prediction = zeros(np,2);
        for j = 1:np
            [prediction(j,2), prediction(j,1)] = findMaximum(final_score(:,:,j));
        end

        for j = 1:np
            if(makeFigure)
                final_score(:,:,j) = final_score(:,:,j)/length(scale_search);
                imToShow = single(oriImg)/255 * 0.5 + mat2im(final_score(:,:,j), jet(100), [0 1])/2;
                imToShow = insertShape(imToShow, 'FilledCircle', [prediction(j,:) 10], 'Color', 'w');
                imToShow = insertShape(imToShow, 'FilledCircle', [gt_joint(j,1:2) 10], 'Color', 'g');
                imToShow = insertShape(imToShow, 'FilledRectangle', [center 15 15], 'Color', 'c');
                
                imshow(imToShow);
                pause;
            end
            
            normsize = norm(prediction(1,:) - prediction(8,:)); % lsho and rhip, following MODEC
       
            error_dist = norm(prediction(j,:) - gt_joint(j,1:2));
            hit = error_dist <= normsize*0.2;

            obj(j) = obj(j) + 1;
            if(hit)
                detected(j) = detected(j) + 1;
            end
            fprintf(' %d ', hit);
        end

        for j = 1:np
            fprintf(' %.3f ', detected(j)/obj(j));
        end
        fprintf(' |%.4f\n', sum(detected)/sum(obj));

        prediction_all(:,:,i) = prediction;
    end

    prediction_file = sprintf('predicts/FLIC_prediction_model_%s.mat', model.description_short);
    save(prediction_file, 'prediction_all');
end

function img_out = preprocess(img, mean, center_map)
    img = double(img) / 256;
    
    img_out = double(img) - mean;
    img_out = permute(img_out, [2 1 3]);
    img_out = img_out(:,:,[3 2 1]);
    img_out(:,:,4) = center_map{1};
    
function scores = applyDNN(images, net)
    input_data = {single(images)};
    % do forward pass to get scores
    % scores are now Width x Height x Channels x Num
    scores = net.forward(input_data);
    scores = scores{1};
    
function [img_padded, pad] = padAround(img, boxsize, center)
    center = round(center);
    h = size(img, 1);
    w = size(img, 2);
    pad(1) = boxsize/2 - center(2); % up
    pad(3) = boxsize/2 - (h-center(2)); % down
    pad(2) = boxsize/2 - center(1); % left
    pad(4) = boxsize/2 - (w-center(1)); % right
    
    pad_up = repmat(img(1,:,:)*0, [pad(1) 1 1])+128;
    img_padded = [pad_up; img];
    pad_left = repmat(img_padded(:,1,:)*0, [1 pad(2) 1])+128;
    img_padded = [pad_left img_padded];
    pad_down = repmat(img_padded(end,:,:)*0, [pad(3) 1 1])+128;
    img_padded = [img_padded; pad_down];
    pad_right = repmat(img_padded(:,end,:)*0, [1 pad(4) 1])+128;
    img_padded = [img_padded pad_right];
    
    center = center + [max(0,pad(2)) max(0,pad(1))];
    img_padded = img_padded(center(2)-(boxsize/2-1):center(2)+boxsize/2, center(1)-(boxsize/2-1):center(1)+boxsize/2, :); %cropping if needed

function [x,y] = findMaximum(map)
    [~,i] = max(map(:));
    [x,y] = ind2sub(size(map), i);
    
function score = resizeIntoScaledImg(score, pad)
    np = size(score,3)-1;
    score = permute(score, [2 1 3]);
    if(pad(1) < 0)
        padup = cat(3, zeros(-pad(1), size(score,2), np), ones(-pad(1), size(score,2), 1));
        score = [padup; score]; % pad up
    else
        score(1:pad(1),:,:) = []; % crop up
    end
    
    if(pad(2) < 0)
        padleft = cat(3, zeros(size(score,1), -pad(2), np), ones(size(score,1), -pad(2), 1));
        score = [padleft score]; % pad left
    else
        score(:,1:pad(2),:) = []; % crop left
    end
    
    if(pad(3) < 0)
        paddown = cat(3, zeros(-pad(3), size(score,2), np), ones(-pad(3), size(score,2), 1));
        score = [score; paddown]; % pad down
    else
        score(end-pad(3)+1:end, :, :) = []; % crop down
    end
    
    if(pad(4) < 0)
        padright = cat(3, zeros(size(score,1), -pad(4), np), ones(size(score,1), -pad(4), 1));
        score = [score padright]; % pad right
    else
        score(:,end-pad(4)+1:end, :) = []; % crop right
    end
    score = permute(score, [2 1 3]);

% function headSize = util_get_head_size(rect)
%     SC_BIAS = 0.6; % 0.8*0.75
%     headSize = SC_BIAS * norm([rect.x2 rect.y2] - [rect.x1 rect.y1]);
    
function bodysize = util_get_bodysize_size(rect)
    bodysize = norm(rect(10,:) - rect(3,:)); % following evalLSP_official

function label = produceCenterLabelMap(im_size, x, y, sigma) %this function is only for center map in testing
    [X,Y] = meshgrid(1:im_size(1), 1:im_size(2));
    X = X - x;
    Y = Y - y;
    D2 = X.^2 + Y.^2;
    Exponent = D2 ./ 2.0 ./ sigma ./ sigma;
    label{1} = exp(-Exponent);