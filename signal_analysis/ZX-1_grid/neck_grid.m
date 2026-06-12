% ============================================================
% Rendered neck grids (concept figure)
% Generates a simple front-view neck outline with 9 grid positions
% using your acquisition numbering:
%
%   3   4   2
%   5   1   6
%   7   8   9
%
% Save as:
%   figure_rendered_neck_grids.m
% ============================================================

clear; clc; close all;

fig = figure('Color','w','Position',[100 100 700 900]);
hold on; axis equal; axis off;

%% ---------------- Neck / lower-face outline ----------------
% Simple symmetric outline for concept rendering
% x right positive, y upward

% jaw / lower face
jaw_x = [-2.8 -2.4 -2.0 -1.6 -1.2 -0.8 -0.4 0 0.4 0.8 1.2 1.6 2.0 2.4 2.8];
jaw_y = [ 7.8  8.4  8.9  9.2  9.45 9.6  9.7  9.75 9.7  9.6  9.45 9.2  8.9  8.4  7.8];

% neck outline
left_neck_x  = [-2.2 -2.0 -1.85 -1.75 -1.65 -1.55 -1.45 -1.38 -1.33];
left_neck_y  = [ 7.4  6.6  5.8   5.0   4.2   3.4   2.6   1.8   1.0];

right_neck_x = -fliplr(left_neck_x);
right_neck_y = fliplr(left_neck_y);

% shoulders / lower end
shoulder_x = [-3.0 -2.6 -2.2 -1.8 -1.2 -0.6 0 0.6 1.2 1.8 2.2 2.6 3.0];
shoulder_y = [ 0.2  0.05 0.0  0.0  0.05 0.1 0.15 0.1 0.05 0.0 0.0 0.05 0.2];

plot(jaw_x, jaw_y, 'k', 'LineWidth', 2.0);
plot(left_neck_x, left_neck_y, 'k', 'LineWidth', 2.0);
plot(right_neck_x, right_neck_y, 'k', 'LineWidth', 2.0);
plot(shoulder_x, shoulder_y, 'k', 'LineWidth', 2.0);

% center line
plot([0 0], [1.0 7.4], '--', 'Color', [0.75 0.75 0.75], 'LineWidth', 1);

%% ---------------- Grid positions ----------------
% 3x3 spatial grid on neck
% top row, middle row, bottom row
y_top = 6.1;
y_mid = 4.3;
y_bot = 2.2;

x_left  = -1.15;
x_mid   = 0.0;
x_right = 1.15;

% numbering according to your real acquisition map:
%   3 4 2
%   5 1 6
%   7 8 9
pos = struct();

pos(1).num = 3; pos(1).xy = [x_left,  y_top];
pos(2).num = 4; pos(2).xy = [x_mid,   y_top];
pos(3).num = 2; pos(3).xy = [x_right, y_top];

pos(4).num = 5; pos(4).xy = [x_left,  y_mid];
pos(5).num = 1; pos(5).xy = [x_mid,   y_mid];
pos(6).num = 6; pos(6).xy = [x_right, y_mid];

pos(7).num = 7; pos(7).xy = [x_left,  y_bot];
pos(8).num = 8; pos(8).xy = [x_mid,   y_bot];
pos(9).num = 9; pos(9).xy = [x_right, y_bot];

% faint grid guides
plot([x_left x_left],   [y_bot y_top], ':', 'Color', [0.75 0.75 0.75], 'LineWidth', 1.2);
plot([x_mid x_mid],     [y_bot y_top], ':', 'Color', [0.75 0.75 0.75], 'LineWidth', 1.2);
plot([x_right x_right], [y_bot y_top], ':', 'Color', [0.75 0.75 0.75], 'LineWidth', 1.2);

plot([x_left x_right], [y_top y_top], ':', 'Color', [0.75 0.75 0.75], 'LineWidth', 1.2);
plot([x_left x_right], [y_mid y_mid], ':', 'Color', [0.75 0.75 0.75], 'LineWidth', 1.2);
plot([x_left x_right], [y_bot y_bot], ':', 'Color', [0.75 0.75 0.75], 'LineWidth', 1.2);

% draw nodes
for i = 1:numel(pos)
    x = pos(i).xy(1);
    y = pos(i).xy(2);

    if pos(i).num == 1
        facecol = [0.15 0.15 0.15];   % emphasize central main position
        txtcol  = 'w';
        r = 0.34;
    else
        facecol = [0.88 0.88 0.88];
        txtcol  = 'k';
        r = 0.30;
    end

    rectangle('Position',[x-r y-r 2*r 2*r],...
              'Curvature',[1 1],...
              'FaceColor',facecol,...
              'EdgeColor','k',...
              'LineWidth',1.5);

    text(x, y, sprintf('%d', pos(i).num),...
        'HorizontalAlignment','center',...
        'VerticalAlignment','middle',...
        'FontWeight','bold',...
        'FontSize',15,...
        'Color',txtcol);
end

%% ---------------- Region labels ----------------
text(0, 10.35, 'Rendered neck grid',...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',18);

text(0, 6.9,  'Top',    'HorizontalAlignment','center', 'FontSize',12, 'Color',[0.35 0.35 0.35]);
text(0, 5.1,  'Middle', 'HorizontalAlignment','center', 'FontSize',12, 'Color',[0.35 0.35 0.35]);
text(0, 3.0,  'Bottom', 'HorizontalAlignment','center', 'FontSize',12, 'Color',[0.35 0.35 0.35]);

text(-2.25, 4.3, 'Lateral',...
    'HorizontalAlignment','center', 'Rotation',90,...
    'FontSize',12, 'Color',[0.35 0.35 0.35]);

text(2.25, 4.3, 'Lateral',...
    'HorizontalAlignment','center', 'Rotation',-90,...
    'FontSize',12, 'Color',[0.35 0.35 0.35]);

text(0, 1.15, 'Central column highlighted by position 1',...
    'HorizontalAlignment','center',...
    'FontSize',11,...
    'Color',[0.3 0.3 0.3]);

xlim([-3.6 3.6]);
ylim([-0.4 10.8]);

%% ---------------- Optional save ----------------
exportgraphics(fig, 'Figure_rendered_neck_grids.png', 'Resolution', 300);