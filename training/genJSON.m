function genJSON(dataset)

    addpath('util/jsonlab/');

    if(strcmp(dataset, 'MPI'))
        mat = load('../dataset/MPI/mpii_human_pose_v1_u12_1/mpii_human_pose_v1_u12_1.mat');
        RELEASE = mat.RELEASE;
        trainIdx = find(RELEASE.img_train);

        % in MPI: (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 
        %          5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 
        %          10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)"  

        tompson = load('../dataset/MPI/Tompson_valid/detections');
        tompson_i_p = [tompson.RELEASE_img_index; tompson.RELEASE_person_index];

        count = 1;
        makeFigure = 0;
        validationCount = 0;

        for i = trainIdx
            numPeople = length(RELEASE.annolist(i).annorect);
            fprintf('image: %d (numPeople: %d) last: %d\n', i, numPeople, trainIdx(end));
            
            for p = 1:numPeople
                loc = find(sum(~bsxfun(@minus, tompson_i_p, [i;p]))==2, 1);
                if(~isempty(loc))
                    validationCount = validationCount + 1;
                    %fprintf('Tomspon''s validation! %d\n', validationCount);
                    isValidation = 1;
                else
                    isValidation = 0;
                end
                joint_all(count).dataset = 'MPI';
                joint_all(count).isValidation = isValidation;
                
                try % sometimes no annotation at all....
                    anno = RELEASE.annolist(i).annorect(p).annopoints.point;
                catch
                    continue;
                end

                % set image path
                %joint_all(count).img_paths = sprintf('dataset/MPI/images/%s', RELEASE.annolist(i).image.name);
                joint_all(count).img_paths = RELEASE.annolist(i).image.name;
                [h,w,~] = size(imread(['../dataset/MPI/images/', joint_all(count).img_paths]));
                joint_all(count).img_width = w;
                joint_all(count).img_height = h;
                joint_all(count).objpos = [RELEASE.annolist(i).annorect(p).objpos.x, RELEASE.annolist(i).annorect(p).objpos.y];
                % set part label: joint_all is (np-3-nTrain)
                
                
                % for this very center person
                for part = 1:length(anno)
                    joint_all(count).joint_self(anno(part).id+1, 1) = anno(part).x;
                    joint_all(count).joint_self(anno(part).id+1, 2) = anno(part).y;
                    try % sometimes no is_visible...
                        if(anno(part).is_visible == 0 || anno(part).is_visible == '0')
                            joint_all(count).joint_self(anno(part).id+1, 3) = 0;
                        else
                            joint_all(count).joint_self(anno(part).id+1, 3) = 1;
                        end
                    catch
                        joint_all(count).joint_self(anno(part).id+1, 3) = 1;
                    end
                end
                % pad it into 16x3
                dim_1 = size(joint_all(count).joint_self, 1);
                dim_3 = size(joint_all(count).joint_self, 3);
                pad_dim = 16 - dim_1;
                joint_all(count).joint_self = [joint_all(count).joint_self; zeros(pad_dim, 3, dim_3)];
                % set scale
                joint_all(count).scale_provided = RELEASE.annolist(i).annorect(p).scale;

                % for other person on the same image
                count_other = 1;
                joint_others = cell(0,0);
                for op = 1:numPeople
                    if(op == p), continue; end
                    try % sometimes no annotation at all....
                        anno = RELEASE.annolist(i).annorect(op).annopoints.point;
                    catch
                        continue;
                    end
                    joint_others{count_other} = zeros(16,3);
                    for part = 1:length(anno)
                        joint_all(count).joint_others{count_other}(anno(part).id+1, 1) = anno(part).x;
                        joint_all(count).joint_others{count_other}(anno(part).id+1, 2) = anno(part).y;
                        try % sometimes no is_visible...
                            if(anno(part).is_visible == 0 || anno(part).is_visible == '0')
                                joint_all(count).joint_others{count_other}(anno(part).id+1, 3) = 0;
                            else
                                joint_all(count).joint_others{count_other}(anno(part).id+1, 3) = 1;
                            end
                        catch
                            joint_all(count).joint_others{count_other}(anno(part).id+1, 3) = 1;
                        end
                    end
                    joint_all(count).scale_provided_other(count_other) = RELEASE.annolist(i).annorect(op).scale;
                    joint_all(count).objpos_other{count_other} = [RELEASE.annolist(i).annorect(op).objpos.x RELEASE.annolist(i).annorect(op).objpos.y];
                    
                    count_other = count_other + 1;
                end
                
                if(makeFigure) % visualizing to debug
                    imshow(joint_all(count).img_paths); 
                    hold on;
                    visiblePart = joint_all(count).joint_self(:,3) == 1;
                    invisiblePart = joint_all(count).joint_self(:,3) == 0;
                    plot(joint_all(count).joint_self(visiblePart, 1), joint_all(count).joint_self(visiblePart,2), 'gx');
                    plot(joint_all(count).joint_self(invisiblePart,1), joint_all(count).joint_self(invisiblePart,2), 'rx');
                    plot(joint_all(count).objpos(1), joint_all(count).objpos(2), 'cs');
                    if(~isempty(joint_all(count).joint_others))
                        for op = 1:size(joint_all(count).joint_others, 3)
                            visiblePart = joint_all(count).joint_others(:,3,op) == 1;
                            invisiblePart = joint_all(count).joint_others(:,3,op) == 0;
                            plot(joint_all(count).joint_others(visiblePart,1,op), joint_all(count).joint_others(visiblePart,2,op), 'mx');
                            plot(joint_all(count).joint_others(invisiblePart,1,op), joint_all(count).joint_others(invisiblePart,2,op), 'cx');
                        end
                    end
                    close all;
                end
                joint_all(count).annolist_index = i;
                joint_all(count).people_index = p;
                joint_all(count).numOtherPeople = length(joint_all(count).joint_others);
                count = count + 1;
                
                %if(count==10), break; endscale_provided
            end
            %if(count==10), break; end
        end
        opt.FileName = 'json/MPI_annotations.json';
        opt.FloatFormat = '%.3f';
        savejson('root', joint_all, opt);
    
    elseif(strcmp(dataset, 'LEEDS'))
        % in cpp: real scale = param_.target_dist()/meta.scale_self = (41/35)/scale_input
        targetDist = 41/35; % in caffe cpp file 41/35
        oriTrTe = load('../dataset/LEEDS/lsp_dataset/joints.mat');
        extTrain = load('../dataset/LEEDS/lspet_dataset/joints.mat');

        % in LEEDS:
        % 1  Right ankle
        % 2  Right knee
        % 3  Right hip
        % 4  Left hip
        % 5  Left knee
        % 6  Left ankle
        % 7  Right wrist
        % 8  Right elbow
        % 9  Right shoulder
        % 10 Left shoulder
        % 11 Left elbow
        % 12 Left wrist
        % 13 Neck
        % 14 Head top
        % 15,16 DUMMY
        % We want to comply to MPII: (1 - r ankle, 2 - r knee, 3 - r hip, 4 - l hip, 5 - l knee, 6 - l ankle, ..
        %                             7 - pelvis, 8 - thorax, 9 - upper neck, 10 - head top, 
        %                             11 - r wrist, 12 - r elbow, 13 - r shoulder, 14 - l shoulder, 15 - l elbow, 16 - l wrist)
        ordering = [1 2 3, 4 5 6, 15 16, 13 14, 7 8 9, 10 11 12]; % should follow MPI 16 parts..?
        oriTrTe.joints(:,[15 16],:) = 0;
        oriTrTe.joints = oriTrTe.joints(:,ordering,:);
        oriTrTe.joints(3,:,:) = 1 - oriTrTe.joints(3,:,:);
        oriTrTe.joints = permute(oriTrTe.joints, [2 1 3]);
        
        extTrain.joints([15 16],:,:) = 0;
        extTrain.joints = extTrain.joints(ordering,:,:);

        count = 1;
       
        path = {'lspet_dataset/images/im%05d.jpg', 'lsp_dataset/images/im%04d.jpg'};
        local_path = {'../dataset/LEEDS/lspet_dataset/images/im%05d.jpg', '../dataset/LEEDS/lsp_dataset/images/im%04d.jpg'};
        num_image = [10000, 1000]; %[10000, 2000];
        
        for dataset = 1:2
            for im = 1:num_image(dataset)
                % trivial stuff for LEEDS
                joint_all(count).dataset = 'LEEDS';
                joint_all(count).isValidation = 0;
                joint_all(count).img_paths = sprintf(path{dataset}, im);
                joint_all(count).numOtherPeople = 0;
                joint_all(count).annolist_index = count;
                joint_all(count).people_index = 1;
                % joints and w, h
                if(dataset == 1)
                    joint_this = extTrain.joints(:,:,im);
                else
                    joint_this = oriTrTe.joints(:,:,im);
                end
                path_this = sprintf(local_path{dataset}, im);
                [h,w,~] = size(imread(path_this));

                joint_all(count).img_width = w;
                joint_all(count).img_height = h;
                joint_all(count).joint_self = joint_this;
                % infer objpos
                invisible = (joint_all(count).joint_self(:,3) == 0);
                if(dataset == 1) %lspet is not tightly cropped
                    joint_all(count).objpos(1) = (min(joint_all(count).joint_self(~invisible, 1)) + max(joint_all(count).joint_self(~invisible, 1))) / 2;
                    joint_all(count).objpos(2) = (min(joint_all(count).joint_self(~invisible, 2)) + max(joint_all(count).joint_self(~invisible, 2))) / 2;
                else
                    joint_all(count).objpos(1) = w/2;
                    joint_all(count).objpos(2) = h/2;
                end
                % visualize
%                 figure(1); clf; imshow(path_this);
%                 hold on;
%                 plot(joint_all(count).joint_self([1 2 3], 1), joint_all(count).joint_self([1 2 3], 2), 'wx', 'Linewidth', 3);
%                 plot(joint_all(count).joint_self([4 5 6], 1), joint_all(count).joint_self([4 5 6], 2), 'bx', 'Linewidth', 3);
%                 plot(joint_all(count).joint_self([9 10], 1), joint_all(count).joint_self([9 10], 2), 'gx', 'Linewidth', 3);
%                 plot(joint_all(count).joint_self([11 12 13], 1), joint_all(count).joint_self([11 12 13], 2), 'mx', 'Linewidth', 3);
%                 plot(joint_all(count).joint_self([14 15 16], 1), joint_all(count).joint_self([14 15 16], 2), 'yx', 'Linewidth', 3);
%                 plot(joint_all(count).joint_self(invisible, 1), joint_all(count).joint_self(invisible, 2), 'rx');
%                 plot(joint_all(count).objpos(1), joint_all(count).objpos(2), 'cs');
                % increase counter and display info
                count = count + 1;
                fprintf('processing %s\n', path_this);
            end
        end

        joint_all = insertMPILikeScale(joint_all, targetDist);
        
        % for i = 1:length(joint_all)
        %     path = ['../dataset/LEEDS/', joint_all(i).img_paths];
        %     figure(1); clf; 
        %     img = imread(path);
        %     scale_in_cpp = targetDist/joint_all(i).scale_provided;
        %     img = imresize(img, scale_in_cpp);
        %     img = [zeros(400,size(img,2),3); img];
        %     img = [zeros(size(img,1),400,3), img];
        %     img = [img; zeros(400,size(img,2),3)];
        %     img = [img, zeros(size(img,1),400,3)];
        %     imshow(img);
        %     hold on;
        %     fprintf('scale: %f\n', scale_in_cpp);
        %     objpos = joint_all(i).objpos * scale_in_cpp + 400;
        %     plot(objpos(1), objpos(2), 'cs');
        %     line([objpos(1)-368/2, objpos(1)-368/2], [objpos(2)+368/2, objpos(2)-368/2], 'lineWidth', 3);
        %     line([objpos(1)+368/2, objpos(1)-368/2], [objpos(2)+368/2, objpos(2)+368/2], 'lineWidth', 3);
        %     line([objpos(1)+368/2, objpos(1)+368/2], [objpos(2)-368/2, objpos(2)+368/2], 'lineWidth', 3);
        %     line([objpos(1)-368/2, objpos(1)+368/2], [objpos(2)-368/2, objpos(2)-368/2], 'lineWidth', 3);
        %     pause;
        % end

        opt.FileName = 'json/LEEDS_annotations.json';
        opt.FloatFormat = '%.3f';
        savejson('root', joint_all, opt);
        
    elseif(strcmp(dataset, 'FLIC'))
        % note FLIC is OC
        targetDist = 41/35;
        constant = 0.0110;
        annotation = load('../dataset/FLIC/examples.mat');
        count_flip = 0;
        anno = {annotation.examples.filepath};
        [~,~,IC] = unique(anno);
        
        count = 1;
        for i = 1:length(annotation.examples)
            if(annotation.examples(i).istest)
                continue;
            end
            fprintf('FLIC images %d, conut = %d\n', i, count);
            joint_all(count).dataset = 'FLIC';
            joint_all(count).isValidation = 0;
            
            img_path = sprintf('images/%s', annotation.examples(i).filepath);
            joint_all(count).img_paths = img_path;
            
            current_IC = IC(i);
            fellow_idx = find(IC == current_IC);
            fellow_idx = setdiff(fellow_idx, i);
            
            joint_all(count).numOtherPeople = length(fellow_idx);
            joint_all(count).annolist_index = i;
            joint_all(count).people_index = 1;
            
            coords = annotation.examples(i).coords;
            coords(isnan(coords)) = [];
            coords = reshape(coords, [2 11]);
            coords(:,[9 10]) = [];
            coords = coords';
            coords = [coords, ones(9,1)];
            
            if(coords(4,1) > coords(1,1))
                coords([1 2 3 4 5 6 7 8],:) = coords([4 5 6 1 2 3 8 7],:);
                count_flip = count_flip + 1;
            end
            
            joint_all(count).joint_self = coords;
            
            img = imread(sprintf('../dataset/FLIC/%s', img_path));
            
            for k = 1:length(fellow_idx)
                coords_fellow = annotation.examples(fellow_idx(k)).coords;
                coords_fellow(isnan(coords_fellow)) = [];
                coords_fellow = reshape(coords_fellow, [2 11]);
                coords_fellow(:,[9 10]) = [];
                coords_fellow = coords_fellow';
                coords_fellow = [coords_fellow, ones(9,1)];
                if(coords_fellow(4,1) > coords_fellow(1,1))
                    coords_fellow([1 2 3 4 5 6 7 8],:) = coords_fellow([4 5 6 1 2 3 8 7],:);
                end
                box = annotation.examples(fellow_idx(k)).torsobox;
                torso_height = box(4) - box(2);
                scale = constant*torso_height;
                
                joint_all(count).joint_others{k} = coords_fellow;
                joint_all(count).scale_provided_other(k) = scale;
                joint_all(count).objpos_other{k} = [(box(1)+box(3))/2, (box(2)*0.7 + box(4)*0.3)];
            end
            
            joint_all(count).img_width = size(img, 2);
            joint_all(count).img_height = size(img, 1);
            
%             imshow(img); hold on;
%             plot(coords(:,1), coords(:,2), 'bx');
%             plot(coords(1:3,1), coords(1:3,2), 'rx');
%             plot(coords(end,1), coords(end,2), 'yx');
            box = annotation.examples(i).torsobox;
%             plot([box(1) box(1) box(3) box(3) box(1)], [box(2) box(4) box(4) box(2) box(2)]);
            torso_height = box(4) - box(2);
            
            joint_all(count).objpos = [(box(1)+box(3))/2, (box(2)*0.7 + box(4)*0.3)];
            
            % in cpp: real scale = param_.target_dist()/meta.scale_self = (41/35)/scale_input
            
            scale = constant * torso_height;
%             real_scale = targetDist/scale;
%             img = imresize(img, real_scale);
%             
%             x = joint_all(i).objpos(1)*real_scale;
%             y = joint_all(i).objpos(2)*real_scale;
%             if(length(fellow_idx)~=0)
%                 clf; imshow(img); hold on;
%                 plot(joint_all(i).objpos(1)*real_scale, joint_all(i).objpos(2)*real_scale, 'cs');
%                 plot([x+184, x+184, x-184, x-184, x+184], [y+184 y-184 y-184 y+184 y+184], 'b-', 'LineWidth', 3);
%                 plot(coords(:,1)*real_scale, coords(:,2)*real_scale, 'rx');
%                 for k = 1:length(fellow_idx)
%                     plot(joint_all(i).objpos_other(k,1)*real_scale, joint_all(i).objpos_other(k,2)*real_scale, 'cs');
%                     coords = joint_all(i).joint_others(:,:,k);
%                     plot(coords(:,1)*real_scale, coords(:,2)*real_scale, 'yx');
%                 end
%             end
             
%             lm(i) = min(coords(:,1)*real_scale) - (x-184);
%             rm(i) = (x+184) - max(coords(:,1)*real_scale);
%             tm(i) = min(coords(:,2)*real_scale) - (y-184);
%             bm(i) = (y+184) - max(coords(:,2)*real_scale);
            
            joint_all(count).scale_provided = scale;
            count = count + 1;
        end
        fprintf('Flipped %d samples\n', count_flip);
        opt.FileName = 'json/FLIC_annotations.json';
        opt.FloatFormat = '%.3f';
        savejson('root', joint_all, opt);
    end
    
function out = parseFile(in)
    % out is like /media/posenas1//141215/141215_Pose1/Kinect//KINECTNODE1-December-15-2014-13-22-33/color/color-2561794.394.png
    % in should be /media/posenas1/Captures/poseMachineKinect/141215/141215_Pose1/Kinect//KINECTNODE1-December-15-2014-13-22-33/color/color-2561794.394.png
    out = strrep(in, '/media/posenas1/', '/media/posenas1/Captures/poseMachineKinect');


function joint_all = insertMPILikeScale(joint_all, targetDist)
    % calculate scales for each image first
    joints = cat(3, joint_all.joint_self);
    joints([7 8],:,:) = [];
    pa = [2 3 7, 5 4 7, 8 0, 10 11 7, 13 12 7];
    x = permute(joints(:,1,:), [3 1 2]);
    y = permute(joints(:,2,:), [3 1 2]);
    vis = permute(joints(:,3,:), [3 1 2]);
    validLimb = 1:14-1;

    x_diff = x(:, [1:7,9:14]) - x(:, pa([1:7,9:14]));
    y_diff = y(:, [1:7,9:14]) - y(:, pa([1:7,9:14]));
    limb_vis = vis(:, [1:7,9:14]) .* vis(:, pa([1:7,9:14]));
    l = sqrt(x_diff.^2 + y_diff.^2);

    for p = 1:14-1 % for each limb. reference: 7th limb, which is 7 to pa(7) (neck to head)
        valid_compare = limb_vis(:,7) .* limb_vis(:,p);
        ratio = l(valid_compare==1, p) ./ l(valid_compare==1, 7);
        r(p) = median(ratio(~isnan(ratio), 1));
    end

    numFiles = size(x_diff, 1);
    all_scales = zeros(numFiles, 1);

    boxSize = 368;
    psize = 64;
    nSqueezed = 0;
    
    for file = 1:numFiles %numFiles
        l_update = l(file, validLimb) ./ r(validLimb);
        l_update = l_update(limb_vis(file,:)==1);
        distToObserve = quantile(l_update, 0.75);
        scale_in_lmdb = distToObserve/35; % can't get too small. 35 is a magic number to balance to MPI
        scale_in_cpp = targetDist/scale_in_lmdb; % can't get too large to be cropped

        visibleParts = joints(:, 3, file);
        visibleParts = joints(visibleParts==1, 1:2, file);
        x_range = max(visibleParts(:,1)) - min(visibleParts(:,1));
        y_range = max(visibleParts(:,2)) - min(visibleParts(:,2));
        scale_x_ub = (boxSize - psize)/x_range;
        scale_y_ub = (boxSize - psize)/y_range;

        scale_shrink = min(min(scale_x_ub, scale_y_ub), scale_in_cpp);
        
        if scale_shrink ~= scale_in_cpp
            nSqueezed = nSqueezed + 1;
            fprintf('img %d: scale = %f %f %f shrink %d\n', file, scale_in_cpp, scale_shrink, min(scale_x_ub, scale_y_ub), nSqueezed);
        else
            fprintf('img %d: scale = %f %f %f\n', file, scale_in_cpp, scale_shrink, min(scale_x_ub, scale_y_ub));
        end
        
        joint_all(file).scale_provided = targetDist/scale_shrink; % back to lmdb unit
    end
    
    fprintf('total %d squeezed!\n', nSqueezed);
