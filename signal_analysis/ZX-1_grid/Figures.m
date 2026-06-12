clear; clc; close all;

%% ============================================================
% Report figure script
% Figure 5: Sensor analysis
%
% Plots:
%   1) Histogram PDE (LED on/off)
%   2) Contrast examples: DC and 500 Hz modulation (arm static/pressing)
%   3) Contrast examples: central and lateral regions (jaw and 'a')
%   4) Display examples: top, middle, bottom; superpose central/lateral (speech)
%   5) ICA on static and speaking signals
%   6) Channels scoring map
%
% Minimal text output, report-oriented figures
%% ============================================================

outdir = 'report_figures';
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

%% ===================== CONFIG =====================
% -------- Figure 1: LED on/off histogram PDE --------
cfg.dc_led_off = "DC_dark_led_off_rep2.mat";
cfg.dc_led_on  = "DC_ambient_led_on_rep2.mat";

% -------- Figure 2: DC vs 500 Hz modulation, arm static/pressing --------
cfg.arm_dc_static = "DC_arm_static_rep3.mat";
cfg.arm_dc_press  = "DC_arm_press_rep3.mat";

cfg.arm_ac_static = "first_arm_static_rep3.mat";
cfg.arm_ac_press  = "first_arm_press_rep3.mat";

% -------- Figure 3: central vs lateral, jaw and 'a' --------
% -------- Figure 3: Contrast examples from central (jaw and 'a') --------
cfg.central_jaw = "first_neck_jaw_rep1.mat";
cfg.central_a   = "first_neck_a_rep1.mat";

% -------- Figure 4: top / middle / bottom speech --------
% adjust these according to your layout and chosen representative files
cfg.top_central_speech    = "map_fourth_speech_rep1.mat";
cfg.top_lateral_speech    = "map_third_speech_rep1.mat";

cfg.middle_central_speech = "map_first_speech_rep1.mat";
cfg.middle_lateral_speech = "map_fifth_speech_rep1.mat";

cfg.bottom_central_speech = "map_eighth_speech_rep1.mat";
cfg.bottom_lateral_speech = "map_seventh_speech_rep1.mat";

% -------- Figure 5: ICA --------
% choose one static-like and one speaking-like multi-channel set
% these should be representative files from different positions
cfg.ica_static_files = {
    "map_first_static_rep1.mat"
    "map_fifth_static_rep1.mat"
    "map_eighth_static_rep1.mat"
};

cfg.ica_speech_files = {
    "map_first_speech_rep1.mat"
    "map_fifth_speech_rep1.mat"
    "map_eighth_speech_rep1.mat"
};

% -------- Figure 6: channel score map --------
cfg.score_csv = "T_position_summary.csv";
score_col = "total_score";

% -------- Common settings --------
thr_mode = 'median';
dropout_thr_frac = 0.5;
plot_sec = 10;  % show first N seconds where suitable

%% ============================================================
%% Figure 1: Histogram PDE (LED on/off)
%% ============================================================

if isfile(cfg.dc_led_off) && isfile(cfg.dc_led_on)
    S1 = load(cfg.dc_led_off);
    S2 = load(cfg.dc_led_on);

    x1 = double(S1.sig(:));
    x2 = double(S2.sig(:));

    [bc1, p1] = compute_pde_hist(x1, 60);
    [bc2, p2] = compute_pde_hist(x2, 60);

    figure('Color','w', 'Position', [100 100 900 350]);

    subplot(1,2,1);
    plot(x1, 'LineWidth', 0.8); hold on;
    plot(x2, 'LineWidth', 0.8);
    xlabel('Sample'); ylabel('ADC');
    title('LED off / LED on traces');
    legend('LED off','LED on','Location','best');
    grid on; box off;

    subplot(1,2,2);
    plot(bc1, p1, 'LineWidth', 1.5); hold on;
    plot(bc2, p2, 'LineWidth', 1.5);
    xlabel('ADC value'); ylabel('PDF');
    title('Histogram PDE');
    legend('LED off','LED on','Location','best');
    grid on; box off;

    exportgraphics(gcf, fullfile(outdir, 'Fig5_1_histogram_pde_led_on_off.png'), 'Resolution', 300);
end

%% ============================================================
%% Figure 2: DC vs 500 Hz modulation (arm static / pressing)
%% ============================================================

if isfile(cfg.arm_dc_static) && isfile(cfg.arm_dc_press) &&...
   isfile(cfg.arm_ac_static) && isfile(cfg.arm_ac_press)

    % DC
    A = load(cfg.arm_dc_static);
    B = load(cfg.arm_dc_press);

    x_dc_static = double(A.sig(:));
    x_dc_press  = double(B.sig(:));
    fs_dc_1 = infer_fs_from_loaded(x_dc_static, A);
    fs_dc_2 = infer_fs_from_loaded(x_dc_press, B);

    % AC
    [x_ac_static, t_ac_static, fs_ac_1] = load_and_demod(cfg.arm_ac_static, thr_mode, dropout_thr_frac);
    [x_ac_press,  t_ac_press,  fs_ac_2] = load_and_demod(cfg.arm_ac_press,  thr_mode, dropout_thr_frac);

    t_dc_1 = (0:numel(x_dc_static)-1)/fs_dc_1;
    t_dc_2 = (0:numel(x_dc_press)-1)/fs_dc_2;

    figure('Color','w', 'Position', [100 100 1000 700]);

    subplot(2,2,1);
    plot_limited_time(t_dc_1, x_dc_static, plot_sec); hold on;
    plot_limited_time(t_dc_2, x_dc_press, plot_sec);
    xlabel('Time (s)'); ylabel('ADC');
    title('DC: arm static / pressing');
    legend('static','pressing','Location','best');
    grid on; box off;

    subplot(2,2,3);
    [f1, P1] = compute_welch_psd(x_dc_static, fs_dc_1);
    [f2, P2] = compute_welch_psd(x_dc_press,  fs_dc_2);
    plot(f1, 10*log10(P1+eps), 'LineWidth', 1.2); hold on;
    plot(f2, 10*log10(P2+eps), 'LineWidth', 1.2);
    xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
    title('DC PSD');
    legend('static','pressing','Location','best');
    xlim([0 50]); grid on; box off;

    subplot(2,2,2);
    plot_limited_time(t_ac_static, x_ac_static, plot_sec); hold on;
    plot_limited_time(t_ac_press,  x_ac_press,  plot_sec);
    xlabel('Time (s)'); ylabel('Demod amplitude');
    title('500 Hz modulation: arm static / pressing');
    legend('static','pressing','Location','best');
    grid on; box off;

    subplot(2,2,4);
    [f1a, P1a] = compute_welch_psd(x_ac_static, fs_ac_1);
    [f2a, P2a] = compute_welch_psd(x_ac_press,  fs_ac_2);
    plot(f1a, 10*log10(P1a+eps), 'LineWidth', 1.2); hold on;
    plot(f2a, 10*log10(P2a+eps), 'LineWidth', 1.2);
    xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
    title('500 Hz demodulated PSD');
    legend('static','pressing','Location','best');
    xlim([0 50]); grid on; box off;

    exportgraphics(gcf, fullfile(outdir, 'Fig5_2_dc_vs_500hz_arm.png'), 'Resolution', 300);
end

%% ============================================================
%% Figure 3: Central vs lateral regions (jaw and "a")
%% ============================================================
%%
% ============================================================
%% Figure 3: Contrast examples from central (jaw and 'a')
%%
% ============================================================

ok_jaw = isfile(cfg.central_jaw);
ok_a   = isfile(cfg.central_a);

if ok_jaw || ok_a
    figure('Color','w','Position',[100 100 1000 700]);

    if ok_jaw
        [x1, t1, fs1] = load_and_demod(cfg.central_jaw, thr_mode, dropout_thr_frac);

        subplot(2,2,1);
        plot_limited_time(t1, x1, plot_sec);
        xlabel('Time (s)'); ylabel('Demod amplitude');
        title('Central jaw');
        grid on; box off;

        subplot(2,2,3);
        [f1, P1] = compute_welch_psd(x1, fs1);
        plot(f1, 10*log10(P1+eps), 'LineWidth', 1.2);
        xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
        title('Central jaw PSD');
        xlim([0 50]); grid on; box off;
    end

    if ok_a
        [x2, t2, fs2] = load_and_demod(cfg.central_a, thr_mode, dropout_thr_frac);

        subplot(2,2,2);
        plot_limited_time(t2, x2, plot_sec);
        xlabel('Time (s)'); ylabel('Demod amplitude');
        title('Central ''a''');
        grid on; box off;

        subplot(2,2,4);
        [f2, P2] = compute_welch_psd(x2, fs2);
        plot(f2, 10*log10(P2+eps), 'LineWidth', 1.2);
        xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
        title('Central ''a'' PSD');
        xlim([0 50]); grid on; box off;
    end

    exportgraphics(gcf, fullfile(outdir, 'Fig5_3_central_jaw_a_examples.png'),...
        'Resolution', 300);
end

%% ============================================================
% %% Figure 4: Top / middle / bottom speech; central vs lateral
% %% ============================================================

speech_sets = {
    cfg.top_central_speech,    cfg.top_lateral_speech,    'Top'
    cfg.middle_central_speech, cfg.middle_lateral_speech, 'Middle'
    cfg.bottom_central_speech, cfg.bottom_lateral_speech, 'Bottom'
};

figure('Color','w', 'Position', [100 100 900 900]);

for i = 1:3
    file_c = speech_sets{i,1};
    file_l = speech_sets{i,2};
    label  = speech_sets{i,3};

    if isfile(file_c) && isfile(file_l)
        [xc, tc, fsc] = load_and_demod(file_c, thr_mode, dropout_thr_frac);
        [xl, tl, fsl] = load_and_demod(file_l, thr_mode, dropout_thr_frac);

        subplot(2,3,i);
        plot_limited_time(tc, zscore_safe(xc), plot_sec); hold on;
        plot_limited_time(tl, zscore_safe(xl), plot_sec);
        xlabel('Time (s)'); ylabel('z-score');
        title(sprintf('%s speech: central vs lateral', label));
        legend('central','lateral','Location','best');
        grid on; box off;

        subplot(2,3,i+3);
        [fc, Pc] = compute_welch_psd(xc, fsc);
        [fl, Pl] = compute_welch_psd(xl, fsl);
        plot(fc, 10*log10(Pc+eps), 'LineWidth', 1.2); hold on;
        plot(fl, 10*log10(Pl+eps), 'LineWidth', 1.2);
        xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
        title(sprintf('%s speech PSD', label));
        legend('central','lateral','Location','best');
        xlim([0 300]); grid on; box off;
    end
end

exportgraphics(gcf, fullfile(outdir, 'Fig5_4_top_middle_bottom_speech.png'), 'Resolution', 300);

% %% ============================================================
% %% Figure 5: ICA on static and speaking signals
% %% ============================================================
% 
% ok_static_ica = all(cellfun(@isfile, cfg.ica_static_files));
% ok_speech_ica = all(cellfun(@isfile, cfg.ica_speech_files));
% 
% if ok_static_ica || ok_speech_ica
%     figure('Color','w', 'Position', [100 100 1000 700]);
% 
%     if ok_static_ica
%         [X_static, fs_static, labels_static] = load_multichannel_demod(cfg.ica_static_files, thr_mode, dropout_thr_frac);
%         [icasig_s, A_s] = run_simple_ica(X_static);
%         t_s = (0:size(icasig_s,1)-1)/fs_static;
% 
%         subplot(2,2,1);
%         plot_limited_time(t_s, normalize_columns(X_static), plot_sec);
%         xlabel('Time (s)'); ylabel('Norm amp');
%         title('Static: observed channels');
%         legend(labels_static, 'Location', 'best');
%         grid on; box off;
% 
%         subplot(2,2,2);
%         plot_limited_time(t_s, normalize_columns(icasig_s), plot_sec);
%         xlabel('Time (s)'); ylabel('ICA comp');
%         title('Static: ICA components');
%         grid on; box off;
%     end
% 
%     if ok_speech_ica
%         [X_speech, fs_speech, labels_speech] = load_multichannel_demod(cfg.ica_speech_files, thr_mode, dropout_thr_frac);
%         [icasig_p, A_p] = run_simple_ica(X_speech);
%         t_p = (0:size(icasig_p,1)-1)/fs_speech;
% 
%         subplot(2,2,3);
%         plot_limited_time(t_p, normalize_columns(X_speech), plot_sec);
%         xlabel('Time (s)'); ylabel('Norm amp');
%         title('Speech: observed channels');
%         legend(labels_speech, 'Location', 'best');
%         grid on; box off;
% 
%         subplot(2,2,4);
%         plot_limited_time(t_p, normalize_columns(icasig_p), plot_sec);
%         xlabel('Time (s)'); ylabel('ICA comp');
%         title('Speech: ICA components');
%         grid on; box off;
%     end
% 
%     exportgraphics(gcf, fullfile(outdir, 'Fig5_5_ica_static_speech.png'), 'Resolution', 300);
% end
% 
% %% ============================================================
% %% Figure 6: Channels scoring map
% %% ============================================================
% 
% if isfile(cfg.score_csv)
%     T = readtable(cfg.score_csv);
% 
%     if ismember('position', T.Properties.VariableNames) && ismember(score_col, T.Properties.VariableNames)
%         figure('Color','w', 'Position', [100 100 500 450]);
%         plot_position_heatmap_report(T, score_col);
%         exportgraphics(gcf, fullfile(outdir, 'Fig5_6_channels_scoring_map.png'), 'Resolution', 300);
%     end
% end

%% ============================================================
%% Local functions
%% ============================================================

function [x, t_x, fs_x] = load_and_demod(filename, thr_mode, dropout_thr_frac)
    S = load(filename);
    sig = double(S.sig(:));
    fs_raw = infer_fs_from_loaded(sig, S);

    if isfield(S, 'ref')
        ref = double(S.ref(:));
        thr_ref = choose_ref_threshold(ref, thr_mode);
        [x, t_x, info] = demodulate_phase_trimmed_mean(ref, sig, fs_raw, thr_ref);
        [x, ~, ~] = repair_dropouts(x, dropout_thr_frac);
        fs_x = info.fs_demod;
    else
        x = sig;
        t_x = (0:numel(x)-1)'/fs_raw;
        fs_x = fs_raw;
    end
end

function [X, fs_common, labels] = load_multichannel_demod(filelist, thr_mode, dropout_thr_frac)
    n = numel(filelist);
    Xcell = cell(n,1);
    fs_list = nan(n,1);
    labels = strings(n,1);

    for i = 1:n
        [x, ~, fsx] = load_and_demod(filelist{i}, thr_mode, dropout_thr_frac);
        Xcell{i} = x(:);
        fs_list(i) = fsx;
        labels(i) = string(filelist{i});
    end

    fs_common = min(fs_list);
    L = min(cellfun(@numel, Xcell));

    X = zeros(L, n);
    for i = 1:n
        X(:,i) = Xcell{i}(1:L);
    end
end

function [icasig, A] = run_simple_ica(X)
    % X: N x channels
    X = double(X);
    X = X - mean(X,1,'omitnan');

    try
        [icasig, A, ~] = fastica(X', 'numOfIC', size(X,2));
        icasig = icasig';
    catch
        % fallback to PCA if fastica unavailable
        [coeff, score, ~] = pca(X);
        icasig = score(:,1:min(3,size(score,2)));
        A = coeff;
    end
end

function Y = normalize_columns(X)
    X = double(X);
    Y = X;
    for k = 1:size(X,2)
        x = X(:,k);
        x = x - mean(x,'omitnan');
        s = std(x,'omitnan');
        if s > 0
            Y(:,k) = x / s;
        else
            Y(:,k) = x;
        end
    end
end

function y = zscore_safe(x)
    x = double(x(:));
    x = x - mean(x,'omitnan');
    s = std(x,'omitnan');
    if s > 0
        y = x / s;
    else
        y = x;
    end
end

function plot_limited_time(t, x, max_sec)
    idx = t <= max_sec;
    if isvector(x)
        plot(t(idx), x(idx), 'LineWidth', 1.0);
    else
        plot(t(idx), x(idx,:), 'LineWidth', 1.0);
    end
end

function plot_position_heatmap_report(T, value_col)
    pos_grid = [...
        "third",  "fourth", "second";
        "fifth",  "first",  "sixth";
        "seventh","eighth", "nineth"
    ];

    M = nan(3,3);
    for r = 1:3
        for c = 1:3
            pos = pos_grid(r,c);
            idx = find(string(T.position) == pos, 1);
            if ~isempty(idx)
                M(r,c) = T.(value_col)(idx);
            end
        end
    end

    imagesc(M);
    axis image;
    colormap(gray);
    colorbar;
    set(gca, 'XTick', 1:3, 'YTick', 1:3, 'XTickLabel', [], 'YTickLabel', []);
    title('Channels scoring map');

    vmin = min(M(:), [], 'omitnan');
    vmax = max(M(:), [], 'omitnan');

    for r = 1:3
        for c = 1:3
            val = M(r,c);
            numlabel = pos_to_num(pos_grid(r,c));

            if isnan(val)
                txt = sprintf('%d\nNaN', numlabel);
                txt_color = 'k';
            else
                txt = sprintf('%d\n%.3f', numlabel, val);
                if vmax > vmin
                    vn = (val - vmin) / (vmax - vmin);
                else
                    vn = 0.5;
                end
                if vn > 0.5
                    txt_color = 'k';
                else
                    txt_color = 'w';
                end
            end

            text(c, r, txt,...
                'HorizontalAlignment', 'center',...
                'VerticalAlignment', 'middle',...
                'Color', txt_color,...
                'FontWeight', 'bold',...
                'FontSize', 11);
        end
    end
end

function n = pos_to_num(pos)
    switch string(pos)
        case "first"
            n = 1;
        case "second"
            n = 2;
        case "third"
            n = 3;
        case "fourth"
            n = 4;
        case "fifth"
            n = 5;
        case "sixth"
            n = 6;
        case "seventh"
            n = 7;
        case "eighth"
            n = 8;
        case "nineth"
            n = 9;
        otherwise
            n = NaN;
    end
end

function thr = choose_ref_threshold(ref, mode)
    ref = double(ref(:));
    switch lower(mode)
        case 'median'
            thr = median(ref, 'omitnan');
        case 'midrange'
            thr = (min(ref)+max(ref))/2;
        otherwise
            thr = median(ref, 'omitnan');
    end
end

function [demod_sig, t_demod, out] = demodulate_phase_trimmed_mean(ref, sig, fs, thr_ref)
    ref = double(ref(:));
    sig = double(sig(:));

    ref_bin = ref > thr_ref;
    dref = diff(ref_bin);
    rise_idx = find(dref == 1) + 1;

    if numel(rise_idx) < 2
        error('Not enough rising edges detected.');
    end

    nCycles = numel(rise_idx) - 1;
    demod_sig = nan(nCycles,1);
    t_demod = nan(nCycles,1);
    samples_per_cycle = nan(nCycles,1);
    on_counts = nan(nCycles,1);
    off_counts = nan(nCycles,1);

    for k = 1:nCycles
        i1 = rise_idx(k);
        i2 = rise_idx(k+1) - 1;
        if i2 <= i1
            continue;
        end

        idx_cycle = i1:i2;
        ref_cycle = ref_bin(idx_cycle);
        sig_cycle = sig(idx_cycle);

        on_idx = find(ref_cycle == 1);
        off_idx = find(ref_cycle == 0);

        if isempty(on_idx) || isempty(off_idx)
            continue;
        end

        on_core = trim_phase_indices(on_idx);
        off_core = trim_phase_indices(off_idx);

        if isempty(on_core) || isempty(off_core)
            continue;
        end

        demod_sig(k) = mean(sig_cycle(on_core)) - mean(sig_cycle(off_core));
        t_demod(k) = mean(idx_cycle)/fs;
        samples_per_cycle(k) = numel(idx_cycle);
        on_counts(k) = numel(on_core);
        off_counts(k) = numel(off_core);
    end

    valid = ~isnan(demod_sig) & ~isnan(t_demod);
    demod_sig = demod_sig(valid);
    t_demod = t_demod(valid);
    samples_per_cycle = samples_per_cycle(valid);
    on_counts = on_counts(valid);
    off_counts = off_counts(valid);

    if numel(t_demod) >= 2
        fs_demod = 1 / median(diff(t_demod));
    else
        fs_demod = NaN;
    end

    out.fs_demod = fs_demod;
    out.n_cycles = numel(demod_sig);
    out.samples_per_cycle = samples_per_cycle;
    out.on_counts = on_counts;
    out.off_counts = off_counts;
end

function core_idx = trim_phase_indices(idx)
    n = numel(idx);
    if n >= 5
        core_idx = idx(3:end-2);
    elseif n == 4
        core_idx = idx(2:3);
    elseif n == 3
        core_idx = idx(2);
    else
        core_idx = idx;
    end
end

function [x_clean, bad_idx, dropout_ratio] = repair_dropouts(x, thr_frac)
    x = double(x(:));
    medx = median(x, 'omitnan');
    bad_idx = x < thr_frac * medx;
    dropout_ratio = mean(bad_idx);

    x_clean = x;
    x_clean(bad_idx) = NaN;
    x_clean = fillmissing(x_clean, 'linear');
    x_clean = fillmissing(x_clean, 'nearest');
end

function [bc, p] = compute_pde_hist(x, nbin)
    x = double(x(:));
    [counts, edges] = histcounts(x, nbin, 'Normalization', 'pdf');
    bc = (edges(1:end-1) + edges(2:end))/2;
    p = counts(:);
end

function [f, Pxx] = compute_welch_psd(x, fs)
    x = double(x(:));
    x = x - mean(x, 'omitnan');

    if length(x) < 64
        f = NaN; Pxx = NaN; return;
    end

    wlen = min(2048, max(64, floor(length(x)/4)));
    win = hamming(wlen);
    noverlap = floor(0.5*wlen);
    nfft = max(512, 2^nextpow2(wlen));
    [Pxx, f] = pwelch(x, win, noverlap, nfft, fs);
end

function fs = infer_fs_from_loaded(x, S)
    if isfield(S, 'fs_est') && ~isempty(S.fs_est)
        fs = S.fs_est; return;
    end
    if isfield(S, 'fs') && ~isempty(S.fs)
        fs = S.fs; return;
    end
    if isfield(S, 'meta')
        if isfield(S.meta, 'frame_rate') && ~isempty(S.meta.frame_rate)
            fs = S.meta.frame_rate; return;
        end
        if isfield(S.meta, 'nominal_frame_rate') && ~isempty(S.meta.nominal_frame_rate)
            fs = S.meta.nominal_frame_rate; return;
        end
        if isfield(S.meta, 'duration_s') && ~isempty(S.meta.duration_s)
            fs = numel(x) / S.meta.duration_s; return;
        end
    end
    if isfield(S, 'duration_s') && ~isempty(S.duration_s)
        fs = numel(x) / S.duration_s; return;
    end
    error('Could not infer fs.');
end