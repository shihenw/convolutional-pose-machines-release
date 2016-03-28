function test_realtest(RELEASE, iteration, scale_search, gpu_device_id)

addpath /usr0/home/shihenw;
addpath('jsonlab/');

caffe('set_device', gpu_device_id);
caffe('reset');

RELEASE_img_index = find(~RELEASE.img_train);
boxsize = 368;
np = 14;
targetDist = 41/35;
batch_size = 1;

deployFile = 'pose_exp_caffe/pose_exp69.3_alltrain_LEEDS/superaug_noclahe/pose_deploy.prototxt';
caffemodel = sprintf('/media/posenas1/Users/shihenw/caffe_model/MPI/pose_exp69.3_alltrain_LEEDS/superaug_noclahe/pose_iter_%d.caffemodel', iteration);
system(sprintf('sed -i "2s/.*/input_dim: %d/" %s', batch_size, deployFile));
system(sprintf('sed -i "4s/.*/input_dim: %d/" %s', boxsize, deployFile));
system(sprintf('sed -i "5s/.*/input_dim: %d/" %s', boxsize, deployFile));
order_to_Tompson = [8 7 11 10 9 12 13 14 3 2 1 4 5 6];
order_to_MPI = [0 1 2 3 4 5 8 9 10 11 12 13 14 15];

makeFigure = 1;
RELEASE_predicted = RELEASE;

center_map = produceCenterLabelMap([boxsize boxsize], boxsize/2, boxsize/2);
count = 0;

% Produce a variable 'keypointsAll', which is prediction
for i = 1:length(RELEASE_img_index)
    %fprintf('%d/%d\n', i, length(RELEASE_img_index));
    img_index = RELEASE_img_index(i);
    
    for person_index = 1:length(RELEASE.annolist(img_index).annorect)   %RELEASE_person_index(i);

        count = count + 1;
        fprintf('%d/%d: %d %d count: %d\n', i, length(RELEASE_img_index), img_index, person_index, count);

        imagePath = RELEASE.annolist(img_index).image.name;
        imagePath = ['images/' imagePath];

        % load image, objpos, and scale, but might fail
        try
            oriImg = imread(imagePath);
            objpos = RELEASE.annolist(img_index).annorect(person_index).objpos;
            center = [objpos.x objpos.y];
            scale_provided = RELEASE.annolist(img_index).annorect(person_index).scale;
            scale0 = targetDist/scale_provided;
        catch
            fprintf('!!!!!!!!!!!!!!!!!!skipped at img_index %d and person_index %d\n', i, person_index);
            count = count - 1;
            continue;
        end
        
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

            matcaffe_init(1, deployFile, caffemodel, 0);
            score{m} = applyDNN(imageToTest);
            
            pool_time = size(imageToTest,1) / size(score{m},1);
            % post-processing the heatmap
            score{m} = imresize(score{m}, pool_time);
            score{m} = resizeIntoScaledImg(score{m}, pad{m});
            score{m} = imresize(score{m}, [size(oriImg,2) size(oriImg,1)]);
        end

        % summing up scores
        final_score = zeros(size(score{1,1}));
        for m = 1:size(score,2)
            final_score = final_score + score{m};
        end
        final_score = permute(final_score, [2 1 3]); 
            
        % ----- generate prediction -----
        prediction = zeros(np,2);
        for j = 1:np
            [prediction(j,2), prediction(j,1)] = findMaximum(final_score(:,:,j));
        end
        prediction(order_to_Tompson,:) = prediction;

        for j = 1:np
            RELEASE_predicted.annolist(img_index).annorect(person_index).annopoints.point(j).id = order_to_MPI(j);
            RELEASE_predicted.annolist(img_index).annorect(person_index).annopoints.point(j).x = prediction(j,1);
            RELEASE_predicted.annolist(img_index).annorect(person_index).annopoints.point(j).y = prediction(j,2);
        end

        final_score(:,:,order_to_Tompson) = final_score(:,:,1:14);
        keypointsAll{i} = prediction;
        
        for j = 1:np
            if(makeFigure)
                peakValue = max(max(final_score(:,:,j)));
                imToShow = single(oriImg)/255 * 0.5 + mat2im(final_score(:,:,j),jet(100),[0 peakValue])/2;
                imToShow = insertShape(imToShow, 'FilledCircle', [prediction(j,:) 10], 'Color', 'w'); 
                %imToShow = insertShape(imToShow, 'FilledCircle', [gt.keypointsAll{i}(j,:) 10], 'Color', 'm'); 
                %imToShow = insertShape(imToShow, 'FilledCircle', [joint_gt(j,1:2,i) 10], 'Color', 'g'); 
                imToShow = insertShape(imToShow, 'FilledRectangle', [center 15 15], 'Color', 'c');
                box_in_scale = boxsize / (scale0);
                imToShow = insertShape(imToShow, 'Rectangle', [center(1)-box_in_scale/2, center(2)-box_in_scale/2, box_in_scale, box_in_scale], 'Color', 'g');
                box_in_scale = boxsize / (scale0 * multiplier(1));
                imToShow = insertShape(imToShow, 'Rectangle', [center(1)-box_in_scale/2, center(2)-box_in_scale/2, box_in_scale, box_in_scale], 'Color', 'c');
                box_in_scale = boxsize / (scale0 * multiplier(end));
                imToShow = insertShape(imToShow, 'Rectangle', [center(1)-box_in_scale/2, center(2)-box_in_scale/2, box_in_scale, box_in_scale], 'Color', 'm');
                %imToShow = insertText(imToShow, [0 0], sprintf('vis: %d', joint_gt(j,3,i)), 'FontSize', 18);
                %imToShow = insertText(imToShow, [0 30], sprintf('Tomp: (%.0f,%.0f)', gt.keypointsAll{i}(j,1), gt.keypointsAll{i}(j,2)), ... 
                %                                             'FontSize', 18);
                %imToShow = insertText(imToShow, [0 60], sprintf('g.t.: (%.0f,%.0f)', joint_gt(j,1,i), joint_gt(j,2,i)), ... 
                %                                             'FontSize', 18);
                imToShow = insertText(imToShow, [0 0], sprintf('pred: (%.0f,%.0f)', prediction(j,1), prediction(j,2)), ... 
                                                             'FontSize', 18);
            
                imwrite(imToShow, sprintf('website_test_CVPR_BGR_scale_exp69.3_LEEDS/test_img_%04d_part_%02d.jpg', count, j));
            end
        end
    end

    if(mod(i,100)==2)
        save(sprintf('mat/MPI_prediction_CVPR_BGR_scale_exp69.3_LEEDS_%05d_%d.mat', i, iteration), 'RELEASE_predicted');
    end
end

save(sprintf('mat/MPI_prediction_CVPR_BGR_scale_exp69.3_LEEDS_%d.mat', iteration), 'RELEASE_predicted');
%save('mat/detections_wei.mat', 'keypointsAll', 'annolist', 'RELEASE_img_index', 'RELEASE_person_index', 'output_joints', 'type');
%save('mat/test_imgPath.mat', 'test_imgPath');

function img_out = preprocess(img, mean, center_map)
    img = double(img) / 256;
    
    img_out = double(img) - mean;
    img_out = permute(img_out, [2 1 3]);
    img_out = img_out(:,:,[3 2 1]); % BGR for opencv training in caffe !!!!!
    img_out(:,:,4) = center_map{1};
    
function scores = applyDNN(images)
    input_data = {single(images)};
    % do forward pass to get scores
    % scores are now Width x Height x Channels x Num
    s_vec = caffe('forward', input_data);
    scores = s_vec{1};
    
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

function headSize = util_get_head_size(rect)
    SC_BIAS = 0.6; % 0.8*0.75
    headSize = SC_BIAS * norm([rect.x2 rect.y2] - [rect.x1 rect.y1]);
    
function label = produceCenterLabelMap(im_size, x, y) %this function is only for center map in testing
    sigma = 21;
    %label{1} = zeros(im_size(1), im_size(2));
    [X,Y] = meshgrid(1:im_size(1), 1:im_size(2));
    X = X - x;
    Y = Y - y;
    D2 = X.^2 + Y.^2;
    Exponent = D2 ./ 2.0 ./ sigma ./ sigma;
    label{1} = exp(-Exponent);