clear
clc

%% 1. 重组数据，使用 rel_path 作为唯一标识
T = readtable("framewise_mode_coefficients.csv\framewise_mode_coefficients.csv");
relPaths = T.rel_path;
[uniqueClips, ~, clipID] = unique(relPaths, 'stable');
nClips = length(uniqueClips);
fprintf('Total unique clips: %d\n', nClips);

% 提取模式数据（假设第9到48列为 mode_1 至 mode_40）
modeData = T{:, 9:48};

% 初始化
clipCell = cell(nClips, 1);
clipSubject = cell(nClips, 1);
clipWord = cell(nClips, 1);
clipNframes = zeros(nClips, 1);

for i = 1:nClips
    idx = (clipID == i);
    % 按帧索引排序
    [~, sortOrd] = sort(T.frame_idx(idx));
    rows = find(idx);
    rows = rows(sortOrd);
    clipCell{i} = modeData(rows, :);
    clipSubject{i} = T.subject{rows(1)};
    clipWord{i} = T.word{rows(1)};
    clipNframes(i) = T.n_frames_in_clip(rows(1));
end

% 确认总数
assert(nClips == 400, 'Clip count not 400, check data loading.');
%% 2. 挑选两个完整组
groupKey = strcat(clipSubject, '_', clipWord);
[uniqueGroups, ~, groupID] = unique(groupKey, 'stable');

% 统计每组 clip 数
groupCounts = accumarray(groupID, 1);
T_group = table(uniqueGroups, groupCounts);
disp(T_group);

% 选择我们要的两个组
targetGroups = {'Subj1_Banana', 'Subj1_engineering'};
groupIdx = zeros(1,2);
for k = 1:2
    idx = find(strcmp(uniqueGroups, targetGroups{k}));
    if ~isempty(idx)
        groupIdx(k) = idx;
    else
        error('Group %s not found.', targetGroups{k});
    end
end

% 提取这两个组的 clip 索引
clipIndices1 = find(groupID == groupIdx(1));
clipIndices2 = find(groupID == groupIdx(2));
fprintf('Group 1 (%s): %d clips\n', targetGroups{1}, length(clipIndices1));
fprintf('Group 2 (%s): %d clips\n', targetGroups{2}, length(clipIndices2));
%% 3. 确定重采样目标长度
% 使用两个组内所有 clips 帧数的中位数
allLengths = [clipNframes(clipIndices1); clipNframes(clipIndices2)];
targetLen = round(median(allLengths));
fprintf('Target resample length: %d frames\n', targetLen);
%% 4. 处理两个组，计算代表信号和组内平均相关系数
nModes = 40;

% 存储结果
repSignal1 = zeros(targetLen, nModes);
repSignal2 = zeros(targetLen, nModes);
intraCorr1 = 0;
intraCorr2 = 0;

% ---- 组 1 ----
nClips1 = length(clipIndices1);
resampled1 = zeros(targetLen, nModes, nClips1);
for i = 1:nClips1
    X = clipCell{clipIndices1(i)};
    t_orig = linspace(0, 1, size(X,1));
    t_new = linspace(0, 1, targetLen);
    for dim = 1:nModes
        resampled1(:, dim, i) = interp1(t_orig, X(:,dim), t_new, 'spline');
    end
end
repSignal1 = mean(resampled1, 3);

% 计算组内 pairwise 相关系数（展平后）
flat1 = reshape(resampled1, targetLen*nModes, nClips1)';
corrMat1 = corr(flat1, 'Type', 'Pearson');
% 取上三角非对角线均值
mask = triu(true(nClips1), 1);
intraCorr1 = mean(corrMat1(mask));
fprintf('Group %s: mean intra-group correlation = %.4f\n', targetGroups{1}, intraCorr1);

% ---- 组 2 ----
nClips2 = length(clipIndices2);
resampled2 = zeros(targetLen, nModes, nClips2);
for i = 1:nClips2
    X = clipCell{clipIndices2(i)};
    t_orig = linspace(0, 1, size(X,1));
    t_new = linspace(0, 1, targetLen);
    for dim = 1:nModes
        resampled2(:, dim, i) = interp1(t_orig, X(:,dim), t_new, 'spline');
    end
end
repSignal2 = mean(resampled2, 3);

flat2 = reshape(resampled2, targetLen*nModes, nClips2)';
corrMat2 = corr(flat2, 'Type', 'Pearson');
mask = triu(true(nClips2), 1);
intraCorr2 = mean(corrMat2(mask));
fprintf('Group %s: mean intra-group correlation = %.4f\n', targetGroups{2}, intraCorr2);
%% 5. 可视化：组1 所有重采样后的 Mode 1 与平均信号
figure;
subplot(2,1,1);
t_plot = linspace(0, 1, targetLen);
hold on;
for i = 1:nClips1
    plot(t_plot, resampled1(:,1,i), 'Color', [0.7 0.7 0.7]);
end
plot(t_plot, repSignal1(:,1), 'r', 'LineWidth', 2);
title(sprintf('Group %s - Mode 1', targetGroups{1}));
xlabel('Normalized time'); ylabel('Coefficient');

subplot(2,1,2);
hold on;
for i = 1:nClips2
    plot(t_plot, resampled2(:,1,i), 'Color', [0.7 0.7 0.7]);
end
plot(t_plot, repSignal2(:,1), 'r', 'LineWidth', 2);
title(sprintf('Group %s - Mode 1', targetGroups{2}));
xlabel('Normalized time'); ylabel('Coefficient');
%% Step: Compute templates for ALL 20 groups and plot in 5x4 grid
% 前提：变量 clipCell, clipSubject, clipWord, groupID, uniqueGroups 已存在

% 重新统计分组（确保包含所有20个组）
groupKey = strcat(clipSubject, '_', clipWord);
[uniqueGroups, ~, groupID] = unique(groupKey, 'stable');
nGroups = length(uniqueGroups);
fprintf('Total groups found: %d\n', nGroups);

% 确定统一重采样长度（取所有clip帧数的中位数）
allLengths = cellfun(@(x) size(x,1), clipCell);
targetLen = round(median(allLengths));
fprintf('Resample target length: %d frames\n', targetLen);

nModes = 40;  % 我们使用全部40个mode，但只画前3个

% 初始化存储模板
templates = cell(nGroups, 1);  % 每个元素为 [targetLen x nModes]

% 对每个组计算平均模板（重采样后平均）
for g = 1:nGroups
    idxClip = find(groupID == g);
    nClips = length(idxClip);
    resampled = zeros(targetLen, nModes, nClips);
    for i = 1:nClips
        X = clipCell{idxClip(i)};
        t_orig = linspace(0, 1, size(X,1));
        t_new  = linspace(0, 1, targetLen);
        for dim = 1:nModes
            resampled(:, dim, i) = interp1(t_orig, X(:,dim), t_new, 'spline');
        end
    end
    templates{g} = mean(resampled, 3);
end

%% 绘制 5x4 子图 (因为4 subjects × 5 words = 20)
% 按 subject 排序显示（假设 subject 顺序为 Subj1, Subj2, Subj4, Subj6）
% 调整子图顺序：行 = word (5种)，列 = subject (4种)

% 提取 subject 和 word 列表
subjects = unique(clipSubject, 'stable');
words = unique(clipWord, 'stable');
nSubj = length(subjects);
nWords = length(words);

figure('Name', 'Template Signals (First 3 Modes)', 'Position', [100 100 1200 900]);
plotIdx = 1;
for w = 1:nWords
    for s = 1:nSubj
        % 找到对应的组索引
        targetKey = [subjects{s}, '_', words{w}];
        g = find(strcmp(uniqueGroups, targetKey));
        if isempty(g)
            continue;
        end
        subplot(nWords, nSubj, plotIdx);
        hold on;
        t_plot = linspace(0, 1, targetLen);
        % 绘制前3个mode，不同颜色
        plot(t_plot, templates{g}(:,1), 'b-', 'LineWidth', 1.5);
        plot(t_plot, templates{g}(:,2), 'r-', 'LineWidth', 1.5);
        plot(t_plot, templates{g}(:,3), 'g-', 'LineWidth', 1.5);
        plot(t_plot, templates{g}(:,4), 'LineWidth', 1.5);

        hold off;
        title(sprintf('%s %s', subjects{s}, words{w}), 'Interpreter', 'none');
        xlabel('Normalized Time');
        ylabel('Coefficient');
        if s==1 && w==1
            legend({'Mode1','Mode2','Mode3', 'Mode4'}, 'Location', 'best');
        end
        grid on;
        plotIdx = plotIdx + 1;
    end
end
sgtitle('Representative Templates (Resampled to Median Length)');

%% Prepare data and output folder
% 假设已存在：
% - uniqueGroups: 20x1 cell, 格式 'SubjX_WordY'
% - templates: 20x1 cell, 每个元素为 [targetLen x 40] 矩阵
% - subjects: 被试列表，如 {'Subj1','Subj2','Subj4','Subj6'}
% - words: 词汇列表，如 {'Banana','engineering','sports','tomorrow','yesterday'}

% 创建输出文件夹
outputDir = 'mode_correlation_heatmaps';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% 定义正确的排序：按照 subject 和 word 顺序排列条件索引
nSubj = length(subjects);
nWords = length(words);
nCond = nSubj * nWords;  % 应为 20

% 构建排序后的条件索引和标签
condOrder = zeros(nCond, 1);
condLabels = cell(nCond, 1);
idx = 1;
for s = 1:nSubj
    for w = 1:nWords
        targetKey = [subjects{s}, '_', words{w}];
        g = find(strcmp(uniqueGroups, targetKey));
        if ~isempty(g)
            condOrder(idx) = g;
            condLabels{idx} = [subjects{s}, '-', words{w}];
            idx = idx + 1;
        else
            warning('Group %s not found', targetKey);
        end
    end
end
% 仅保留实际找到的组（万一有缺失）
condOrder = condOrder(1:idx-1);
condLabels = condLabels(1:idx-1);
nValidCond = length(condOrder);

% 提取模板并按排序重新排列
templates_sorted = templates(condOrder);  % cell 数组，每个元素 [targetLen x 40]

% 将模板数据转换为 3D 数组便于索引：[nFrames x nModes x nConditions]
targetLen = size(templates_sorted{1}, 1);
nModes = 40;
data3D = zeros(targetLen, nModes, nValidCond);
for c = 1:nValidCond
    data3D(:, :, c) = templates_sorted{c};
end

%% 对40个 mode 生成热图
save_heatmap = false;
nModesToPlot = 40;
for modeIdx = 1:nModesToPlot
    % 提取该 mode 所有条件的时序 (nFrames x nConditions)
    modeData = squeeze(data3D(:, modeIdx, :));  % [nFrames x nCond]
    
    % 计算相关系数矩阵（Pearson）
    R = corr(modeData, 'Type', 'Pearson');
    
    % 取绝对值
    R_abs = abs(R);
    
    % 创建图形
    figure('Visible', 'off');  % 不弹出窗口，直接保存
    imagesc(R_abs);
    colormap('parula');
    colorbar;
    clim([0 1]);               % 相关系数绝对值范围 0~1
    axis square;
    title(sprintf('Mode %d - Absolute Correlation', modeIdx));
    
    % 设置坐标轴标签
    set(gca, 'XTick', 1:nValidCond, 'XTickLabel', condLabels, ...
             'YTick', 1:nValidCond, 'YTickLabel', condLabels, ...
             'XTickLabelRotation', 45, 'FontSize', 8);
    
    % 添加虚线框标注 subject 分块（对角线 5x5 方块）
    hold on;
    for s = 1:nSubj
        startIdx = (s-1)*nWords + 1;
        endIdx   = s*nWords;
        % 画矩形框，LineStyle '--'，颜色灰色
        rectangle('Position', [startIdx-0.5, startIdx-0.5, nWords, nWords], ...
                  'EdgeColor', [0.5 0.5 0.5], 'LineWidth', 1.5, 'LineStyle', '--');
    end
    hold off;
    
    % 保存为 SVG
    if save_heatmap == true
    filename = fullfile(outputDir, sprintf('Mode_%02d_Correlation.svg', modeIdx));
    saveas(gcf, filename, 'svg');
    close(gcf); 
    end
end

fprintf('Heatmaps saved to folder: %s\n', outputDir);

%% --- Supplemental: Average heatmap across all 40 modes (Averaged Template) ---
% 假设平均模板数据已存储为 data3D_avg，尺寸为 [targetLen, 40, nValidCond]
% 若变量名不同，请相应调整

% 预分配累积矩阵
R_avg_sum = zeros(nValidCond, nValidCond);

for mode = 1:40
    modeData = squeeze(data3D(:, mode, :));  % [targetLen x nCond]
    R = abs(corr(modeData, 'Type', 'Pearson'));
    R_avg_sum = R_avg_sum + R;
end

R_avg = R_avg_sum / 40;   % 平均绝对相关系数矩阵

% 绘制汇总热图
figure('Visible', 'off');
imagesc(R_avg);
colormap('parula');
colorbar;
clim([0 1]);
axis square;
title('Averaged Template - Mean Abs Corr Across All 40 Modes');

set(gca, 'XTick', 1:nValidCond, 'XTickLabel', condLabels, ...
         'YTick', 1:nValidCond, 'YTickLabel', condLabels, ...
         'XTickLabelRotation', 45, 'FontSize', 8);

% 添加虚线框（被试分块）
hold on;
nSubj = length(subjects);
nWords = length(words);
for s = 1:nSubj
    startIdx = (s-1)*nWords + 1;
    endIdx = s*nWords;
    rectangle('Position', [startIdx-0.5, startIdx-0.5, nWords, nWords], ...
              'EdgeColor', [0.5 0.5 0.5], 'LineWidth', 1.5, 'LineStyle', '--');
end
hold off;

% 保存为 SVG（与之前平均模板热图放在同一目录或指定目录）
saveas(gcf, fullfile(outputDir, 'AveragedTemplate_MeanAllModes.svg'), 'svg');
close(gcf);

%% ----  Median Representative Template Extraction ----
% 重新确认分组与排序
groupKey = strcat(clipSubject, '_', clipWord);
[uniqueGroups, ~, groupID] = unique(groupKey, 'stable');
nGroups = length(uniqueGroups);

% 重新计算 targetLen (如果不存在)
if ~exist('targetLen', 'var')
    allLengths = cellfun(@(x) size(x,1), clipCell);
    targetLen = round(median(allLengths));
    fprintf('Recalculated targetLen = %d\n', targetLen);
end

nModes = 40;
% 构建排序顺序
condOrder = zeros(nGroups, 1);
condLabels = cell(nGroups, 1);
idx = 1;
for s = 1:length(subjects)
    for w = 1:length(words)
        targetKey = [subjects{s}, '_', words{w}];
        g = find(strcmp(uniqueGroups, targetKey));
        if ~isempty(g)
            condOrder(idx) = g;
            condLabels{idx} = [subjects{s}, '-', words{w}];
            idx = idx + 1;
        end
    end
end
condOrder = condOrder(1:idx-1);
condLabels = condLabels(1:idx-1);
nValidCond = length(condOrder);

% 定义重采样函数
resampleClip = @(X, L) interp1(linspace(0,1,size(X,1)), X, linspace(0,1,L), 'spline');
%% Scheme A: Mode-wise best representative selection
templates_A = cell(nGroups, 1);  % 每个元素 [targetLen x nModes]
bestClipIdx_A = zeros(nGroups, nModes);  % 记录每个条件每个mode选中的clip原始索引

for g = 1:nGroups
    clipIdx = find(groupID == g);
    nClips = length(clipIdx);
    % 重采样所有clip到统一长度
    resampled = zeros(targetLen, nModes, nClips);
    for i = 1:nClips
        X = clipCell{clipIdx(i)};
        resampled(:,:,i) = resampleClip(X, targetLen);
    end
    
    % 对每个mode独立挑选最佳代表
    rep = zeros(targetLen, nModes);
    for mode = 1:nModes
        % 提取该mode所有clip的数据 [nFrames x nClips]
        modeData = squeeze(resampled(:, mode, :));
        % 计算相关系数矩阵
        R = corr(modeData, 'Type', 'Pearson');
        % 计算每个clip与其他clip相关系数的中位数（排除自相关）
        affinity = zeros(nClips, 1);
        for i = 1:nClips
            others = setdiff(1:nClips, i);
            affinity(i) = median(R(i, others));
        end
        % 选择亲和度最高的clip
        [~, bestIdx] = max(affinity);
        rep(:, mode) = modeData(:, bestIdx);
        bestClipIdx_A(g, mode) = clipIdx(bestIdx);
    end
    templates_A{g} = rep;
end
%% Scheme B: Overall best representative clip (flattened across modes)
templates_B = cell(nGroups, 1);
bestClipIdx_B = zeros(nGroups, 1);

for g = 1:nGroups
    clipIdx = find(groupID == g);
    nClips = length(clipIdx);
    % 重采样所有clip
    resampled = zeros(targetLen, nModes, nClips);
    for i = 1:nClips
        X = clipCell{clipIdx(i)};
        resampled(:,:,i) = resampleClip(X, targetLen);
    end
    
    % 展平每个clip为长向量 [targetLen * nModes, 1]
    flatClips = zeros(targetLen * nModes, nClips);
    for i = 1:nClips
        flatClips(:, i) = reshape(resampled(:,:,i), [], 1);
    end
    
    % 计算整体相关系数矩阵
    R = corr(flatClips, 'Type', 'Pearson');
    % 计算亲和度（中位数）
    affinity = zeros(nClips, 1);
    for i = 1:nClips
        others = setdiff(1:nClips, i);
        affinity(i) = median(R(i, others));
    end
    [~, bestIdx] = max(affinity);
    bestClipIdx_B(g) = clipIdx(bestIdx);
    templates_B{g} = resampled(:,:,bestIdx);
end
%% ---- Scheme B: Overall Best Representative Clip (Full Template) ----
% 生成全部 40 个 mode 的热图，每个热图基于 Scheme B 模板计算绝对相关系数。
% 假设工作区已存在：clipCell, clipSubject, clipWord, groupID, uniqueGroups,
% subjects, words, condOrder, condLabels, targetLen 等变量。

save_scheme_B = true;
% 确认目标长度
if ~exist('targetLen', 'var')
    allLengths = cellfun(@(x) size(x,1), clipCell);
    targetLen = round(median(allLengths));
    fprintf('Recalculated targetLen = %d\n', targetLen);
end

nModes = 40;
nGroups = length(uniqueGroups);

% 定义重采样辅助函数
resampleClip = @(X, L) interp1(linspace(0,1,size(X,1)), X, linspace(0,1,L), 'spline');

% 计算 Scheme B 模板
templates_B = cell(nGroups, 1);
fprintf('Computing Scheme B templates for %d groups...\n', nGroups);

for g = 1:nGroups
    clipIdx = find(groupID == g);
    nClips = length(clipIdx);
    
    % 重采样所有 clip 至统一长度
    resampled = zeros(targetLen, nModes, nClips);
    for i = 1:nClips
        X = clipCell{clipIdx(i)};
        resampled(:,:,i) = resampleClip(X, targetLen);
    end
    
    % 展平每个 clip 为长向量 [targetLen * nModes, 1]
    flatClips = zeros(targetLen * nModes, nClips);
    for i = 1:nClips
        flatClips(:, i) = reshape(resampled(:,:,i), [], 1);
    end
    
    % 计算整体相关系数矩阵
    R = corr(flatClips, 'Type', 'Pearson');
    
    % 计算每个 clip 与其他 clips 相关系数的中位数（亲和度）
    affinity = zeros(nClips, 1);
    for i = 1:nClips
        others = setdiff(1:nClips, i);
        affinity(i) = median(R(i, others));
    end
    
    % 选取亲和度最高的 clip 作为该组模板
    [~, bestIdx] = max(affinity);
    templates_B{g} = resampled(:,:,bestIdx);
end

% 按 condOrder 排序并转换为 3D 数组
nValidCond = length(condOrder);
data3D = zeros(targetLen, nModes, nValidCond);
for c = 1:nValidCond
    data3D(:, :, c) = templates_B{condOrder(c)};
end

% 创建输出文件夹
outputDir = 'schemeB_heatmaps';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% 循环生成 40 个 mode 的热图
fprintf('Generating heatmaps for all 40 modes...\n');
for mode = 1:nModes
    modeData = squeeze(data3D(:, mode, :));  % [targetLen x nCond]
    R_abs = abs(corr(modeData, 'Type', 'Pearson'));
    
    figure('Visible', 'off');
    imagesc(R_abs);
    colormap('parula');
    colorbar;
    clim([0 1]);
    axis square;
    title(sprintf('Scheme B - Mode %d (Abs Correlation)', mode));
    
    set(gca, 'XTick', 1:nValidCond, 'XTickLabel', condLabels, ...
             'YTick', 1:nValidCond, 'YTickLabel', condLabels, ...
             'XTickLabelRotation', 45, 'FontSize', 8);
    
    % 添加虚线框标示同一被试的 5 个词块
    hold on;
    nSubj = length(subjects);
    nWords = length(words);
    for s = 1:nSubj
        startIdx = (s-1)*nWords + 1;
        endIdx = s*nWords;
        rectangle('Position', [startIdx-0.5, startIdx-0.5, nWords, nWords], ...
                  'EdgeColor', [0.5 0.5 0.5], 'LineWidth', 1.5, 'LineStyle', '--');
    end
    hold off;
    
    if save_scheme_B == true
    filename = fullfile(outputDir, sprintf('SchemeB_Mode%02d_Corr.svg', mode));
    saveas(gcf, filename, 'svg');
    close(gcf);
    end
end

%% --- Supplemental: Scheme A average heatmap across all 40 modes ---

data3D_A = zeros(targetLen, 40, nValidCond);
for c = 1:nValidCond
    data3D_A(:, :, c) = templates_A{condOrder(c)};
end

% 假设 data3D_A 尺寸为 [targetLen, 40, nValidCond]

R_avg_sum = zeros(nValidCond, nValidCond);

for mode = 1:40
    modeData = squeeze(data3D_A(:, mode, :));  % [targetLen x nCond]
    R = abs(corr(modeData, 'Type', 'Pearson'));
    R_avg_sum = R_avg_sum + R;
end

R_avg = R_avg_sum / 40;

% 绘制汇总热图
figure('Visible', 'off');
imagesc(R_avg);
colormap('parula');
colorbar;
clim([0 1]);
axis square;
title('Scheme A - Mean Abs Corr Across All 40 Modes');

set(gca, 'XTick', 1:nValidCond, 'XTickLabel', condLabels, ...
         'YTick', 1:nValidCond, 'YTickLabel', condLabels, ...
         'XTickLabelRotation', 45, 'FontSize', 8);

% 虚线框标示被试分块
hold on;
nSubj = length(subjects);
nWords = length(words);
for s = 1:nSubj
    startIdx = (s-1)*nWords + 1;
    endIdx = s*nWords;
    rectangle('Position', [startIdx-0.5, startIdx-0.5, nWords, nWords], ...
              'EdgeColor', [0.5 0.5 0.5], 'LineWidth', 1.5, 'LineStyle', '--');
end
hold off;

% 保存至 schemeA_heatmaps 文件夹（或其他指定路径）
outputDir = 'schemeA_heatmaps';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
saveas(gcf, fullfile(outputDir, 'SchemeA_MeanAllModes.svg'), 'svg');
close(gcf);

fprintf('Scheme A mean heatmap saved to %s\n', fullfile(outputDir, 'SchemeA_MeanAllModes.svg'));


%% --- Supplemental: Average heatmap across all 40 modes (Scheme B) ---
% 假设 Scheme B 模板数据已存储为 data3D_B，尺寸为 [targetLen, 40, nValidCond]
% 若变量名不同，请相应调整（例如在之前 Scheme B 代码中使用了 data3D）

R_avg_sum = zeros(nValidCond, nValidCond);

for mode = 1:40
    modeData = squeeze(data3D(:, mode, :));  % 注意：Scheme B 代码中变量为 data3D
    R = abs(corr(modeData, 'Type', 'Pearson'));
    R_avg_sum = R_avg_sum + R;
end

R_avg = R_avg_sum / 40;

% 绘制汇总热图
figure('Visible', 'off');
imagesc(R_avg);
colormap('parula');
colorbar;
clim([0 1]);
axis square;
title('Scheme B - Mean Abs Corr Across All 40 Modes');

set(gca, 'XTick', 1:nValidCond, 'XTickLabel', condLabels, ...
         'YTick', 1:nValidCond, 'YTickLabel', condLabels, ...
         'XTickLabelRotation', 45, 'FontSize', 8);

hold on;
nSubj = length(subjects);
nWords = length(words);
for s = 1:nSubj
    startIdx = (s-1)*nWords + 1;
    endIdx = s*nWords;
    rectangle('Position', [startIdx-0.5, startIdx-0.5, nWords, nWords], ...
              'EdgeColor', [0.5 0.5 0.5], 'LineWidth', 1.5, 'LineStyle', '--');
end
hold off;

% 保存至 schemeB_heatmaps 文件夹
saveas(gcf, fullfile(outputDir, 'SchemeB_MeanAllModes.svg'), 'svg');
close(gcf);

