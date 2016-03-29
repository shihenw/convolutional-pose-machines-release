function visualize(test_image, heatMaps, predict, param, rectangle, interestPart)

model = param.model(param.modelID);
np = model.np;
part_str = model.part_str;
nstage = model.stage;
im = imread(test_image);
facealpha = 0.6; % for limb transparency
[x_start, x_end, y_start, y_end] = get_display_range(rectangle, im);
predict = bsxfun(@minus, predict, [x_start, y_start]); % offset due to display range

%% create figure and axes
hFig = figure(2);
if(np == 14 && nstage == 6)
    num_col = 8;
    ha = tight_subplot(3, num_col, [0.03 0.01], [0.01 0.04], [0.01 0.01]);
    bg = get(gcf, 'Color');
    set(ha(1), 'Color', bg, 'xcolor', bg, 'ycolor', bg);
    axes(ha(1));
    text(0.5, 0.55,{'Convolutional', 'Pose Machines'},'fontsize',20, 'HorizontalAlignment', 'center');
    text(0.5, 0.35,'CVPR 2016','fontsize',14, 'HorizontalAlignment', 'center');
    set(ha(2), 'Color', bg, 'xcolor', bg, 'ycolor', bg);
    axes(ha(2));
    text(0.5, 0.55,'Using model:','fontsize',14, 'HorizontalAlignment', 'center');
    text(0.5, 0.45, model.description_short,'fontsize',14, 'Interpreter', 'none', 'HorizontalAlignment', 'center');
elseif(np == 9 && nstage == 4)
    num_col = 5;
    ha = tight_subplot(3, num_col, [0.03 0.01], [0.01 0.04], [0.01 0.01]);
else
    error('define your own layout!');
end
set(hFig, 'Position', get(0, 'ScreenSize'));

% plot stagewise heatmap for interested part
part_id_C = strfind(part_str, interestPart);
part_id = not(cellfun('isempty', part_id_C));
for s = 1:nstage
    response = heatMaps{s}(:,:, part_id);
    max_value = max(max(response));
    mapIm = mat2im(response, jet(100), [0 max_value*0.8]);
    imToShow = mapIm*0.5 + (single(im)/255)*0.5;
    axes(ha(s+num_col-nstage));
    imshow(imToShow(y_start:y_end, x_start:x_end, :));
    title(sprintf('%s: stage %d', interestPart, s));
end

% plot all parts and background
truncate = zeros(1,np);
for part = 1:np+1
    response = heatMaps{end}(:,:,part);
    max_value = max(max(response));
    mapIm = mat2im(response, jet(100), [0 max_value]);

    imToShow = mapIm*0.5 + (single(im)/255)*0.5;
    axes(ha(num_col+part+1));
    imshow(imToShow(y_start:y_end, x_start:x_end, :));
    
    % plot predictions on heat maps
    if(part~=np+1)
        if(max_value > 0.15)
            title(sprintf('%s (%f)', part_str{part}, max_value));
        else
            title({sprintf('%s (%f)', part_str{part}, max_value), '(truncated?)'});
            truncate(part) = 1;
        end
        hold on;
        plot(predict(part,1), predict(part,2), 'wx', 'LineWidth', 2);
    else
        title('bkg');
    end
end

% plot full pose
axes(ha(num_col+1));
imshow(im(y_start:y_end, x_start:x_end, :));
hold on;
bodyHeight = max(predict(:,2)) - min(predict(:,2));
plot_visible_limbs(model, facealpha, predict, truncate, bodyHeight/30);
plot(predict(:,1), predict(:,2), 'k.', 'MarkerSize', bodyHeight/32);
title('Full Pose');



%% function area
function [x_start, x_end, y_start, y_end] = get_display_range(rectangle, im)
    x_start = max(rectangle(1), 1);
    x_end = min(rectangle(1)+rectangle(3), size(im,2));
    y_start = max(rectangle(2), 1);
    y_end = min(rectangle(2)+rectangle(4), size(im,1));
    center = [(x_start + x_end)/2, (y_start + y_end)/2];
    % enlarge range
    x_start = max(1, round(x_start - (center(1) - x_start) * 0.2));
    x_end = min(size(im,2), round(x_end + (center(1) - x_start) * 0.2));
    y_start = max(1, round(y_start - (center(2) - y_start) * 0.2));
    y_end = min(size(im,1), round(y_end + (center(2) - y_start) * 0.2));

function plot_visible_limbs(model, facealpha, predict, truncate, stickwidth)
    % plot limbs as ellipses
    limbs = model.limbs;
    colors = hsv(length(limbs));

    for p = 1:size(limbs,1) % visibility?
        if(truncate(limbs(p,1))==1 || truncate(limbs(p,2))==1)
            continue;
        end
        X = predict(limbs(p,:),1);
        Y = predict(limbs(p,:),2);

        if(~sum(isnan(X)))
            a = 1/2 * sqrt((X(2)-X(1))^2+(Y(2)-Y(1))^2);
            b = stickwidth;
            t = linspace(0,2*pi);
            XX = a*cos(t);
            YY = b*sin(t);
            w = atan2(Y(2)-Y(1), X(2)-X(1));
            x = (X(1)+X(2))/2 + XX*cos(w) - YY*sin(w);
            y = (Y(1)+Y(2))/2 + XX*sin(w) + YY*cos(w);
            h = patch(x,y,colors(p,:));
            set(h,'FaceAlpha',facealpha);
            set(h,'EdgeAlpha',0);
        end
    end