clear; clc; close all;

%  Expected variables in.mat:
%    sig                      required
%    ref                      optional (for AC/modulated recordings)
%    fs / fs_est / duration_s optional
%
%  Outputs:
%    T_file_features
%    T_position_summary
%% ============================================================

%% ---------------- User settings ----------------
file_pattern = 'map_*.mat';

% Reference threshold mode for AC demodulation:
% 'median'  : thr = median(ref)
% 'midrange': thr = (min(ref)+max(ref))/2
thr_mode = 'median';

% Frequency bands for feature extraction
band_low_1 = [0.5 5];
band_low_2 = [5 50];
band_vib    = [70 300];   % useful if fs_demod supports it
band_wide   = [20 500];

% Weights for final position score
w_static = 0.30;
w_jaw    = 0.30;
w_speech = 0.40;

% Whether to generate some debug plots for first few files
make_debug_plots = false;
max_debug_plots = 3;

%% ---------------- Scan files ----------------
files = dir(file_pattern);
if isempty(files)
    error('No files matched pattern: %s', file_pattern);
end

fprintf('Found %d files.\n', numel(files));

rows = struct([]);
dbg_count = 0;

%% ---------------- Main loop ----------------
for k = 1:numel(files)
    fname = files(k).name;
    fprintf('Processing %s\n', fname);

    S = load(fname);

    % -------- Parse filename --------
    meta = parse_map_filename(fname);

    % Skip unknown task/position if parser failed badly
    if strcmp(meta.position, "unknown")
        warning('Could not parse position from file: %s', fname);
    end
    if strcmp(meta.task, "unknown")
        warning('Could not parse task from file: %s', fname);
    end

    % -------- Check signal --------
    if ~isfield(S, 'sig')
        warning('Skipping %s: no variable "sig".', fname);
        continue;
    end

    sig = double(S.sig(:));
    fs_raw = infer_fs_from_loaded(sig, S);

    isAC = isfield(S, 'ref');

    % -------- Prepare analysis signal x --------
    ref = [];
    demod_info = struct();
    fs_x = fs_raw;

    if isAC
        ref = double(S.ref(:));
        if numel(ref) ~= numel(sig)
            warning('Skipping %s: ref and sig lengths differ.', fname);
            continue;
        end

        thr_ref = choose_ref_threshold(ref, thr_mode);

        try
            [x, t_x, demod_info] = demodulate_phase_trimmed_mean(ref, sig, fs_raw, thr_ref);
        catch ME
            warning('Demod failed for %s: %s', fname, ME.message);
            continue;
        end

        fs_x = demod_info.fs_demod;
        raw_or_demod = "demod";
    else
        x = sig;
        t_x = (0:numel(x)-1)'/fs_x;
        raw_or_demod = "raw";
    end

    if isempty(x) || numel(x) < 20 || isnan(fs_x) || fs_x <= 0
        warning('Skipping %s: invalid analysis signal or fs.', fname);
        continue;
    end

    % -------- Extract task-specific features --------
    feat = extract_features_by_task(x, fs_x, meta.task,...
        band_low_1, band_low_2, band_vib, band_wide);

    % -------- Store row --------
    row = struct();
    row.file = string(fname);
    row.position = string(meta.position);
    row.task = string(meta.task);
    row.rep = meta.rep;
    row.isAC = isAC;
    row.mode = raw_or_demod;
    row.fs_raw = fs_raw;
    row.fs_analysis = fs_x;
    row.N_raw = numel(sig);
    row.N_analysis = numel(x);

    if isAC
        row.thr_ref = thr_ref;
        row.n_cycles = getfield_safe(demod_info, 'n_cycles', NaN);
        row.samples_per_cycle_med = median(getfield_safe(demod_info, 'samples_per_cycle', NaN));
        row.on_counts_med = median(getfield_safe(demod_info, 'on_counts', NaN));
        row.off_counts_med = median(getfield_safe(demod_info, 'off_counts', NaN));
    else
        row.thr_ref = NaN;
        row.n_cycles = NaN;
        row.samples_per_cycle_med = NaN;
        row.on_counts_med = NaN;
        row.off_counts_med = NaN;
    end

    feat_names = fieldnames(feat);
    for i = 1:numel(feat_names)
        row.(feat_names{i}) = feat.(feat_names{i});
    end

    if isempty(rows)
        rows = row;
    else
        rows(end+1) = row; %#ok<SAGROW>
    end

    % -------- Optional debug plots --------
    if make_debug_plots && dbg_count < max_debug_plots
        dbg_count = dbg_count + 1;
        figure('Name', char(fname), 'Color', 'w');
        subplot(3,1,1);
        if isAC
            plot((0:numel(ref)-1)/fs_raw, ref, 'LineWidth', 0.8); hold on;
            yline(thr_ref, '--r', 'thr');
            ylabel('ref');
            title(sprintf('%s | ref', fname), 'Interpreter', 'none');
        else
            plot((0:numel(sig)-1)/fs_raw, sig, 'LineWidth', 0.8);
            ylabel('sig');
            title(sprintf('%s | raw sig', fname), 'Interpreter', 'none');
        end
        grid on;

        subplot(3,1,2);
        plot((0:numel(sig)-1)/fs_raw, sig, 'LineWidth', 0.8);
        ylabel('raw sig');
        grid on;

        subplot(3,1,3);
        plot(t_x, x, 'LineWidth', 0.8);
        xlabel('Time (s)');
        ylabel('analysis sig');
        title(sprintf('Task=%s | Position=%s | fs=%.2f Hz', meta.task, meta.position, fs_x));
        grid on;
    end
end

if isempty(rows)
    error('No valid files were processed.');
end

%% ---------------- File-level table ----------------
T_file_features = struct2table(rows);
disp(' ');
disp('================ FILE-LEVEL FEATURES ================');
disp(T_file_features(:, intersect(...
    ["file","position","task","rep","isAC","fs_analysis","std0","range0","drift0",...
     "band_ratio_vib","dom_freq","autocorr_peak"],...
    string(T_file_features.Properties.VariableNames)) ));

%% ---------------- Position-level aggregation ----------------
positions = unique(T_file_features.position);
pos_rows = struct([]);

for p = 1:numel(positions)
    pos = positions(p);
    Tp = T_file_features(T_file_features.position == pos, :);

    T_static = Tp(Tp.task == "static", :);
    T_jaw    = Tp(Tp.task == "jaw", :);
    T_speech = Tp(Tp.task == "speech", :);

    out = struct();
    out.position = pos;

    % ---------- STATIC summary ----------
    out.n_static = height(T_static);
    out.static_std0_mean   = nanmean_safe(T_static, 'std0');
    out.static_range0_mean = nanmean_safe(T_static, 'range0');
    out.static_drift0_mean = nanmean_safe(T_static, 'drift0');

    % ---------- JAW summary ----------
    out.n_jaw = height(T_jaw);
    out.jaw_std0_mean         = nanmean_safe(T_jaw, 'std0');
    out.jaw_range0_mean       = nanmean_safe(T_jaw, 'range0');
    out.jaw_autocorr_mean     = nanmean_safe(T_jaw, 'autocorr_peak');
    out.jaw_domfreq_mean      = nanmean_safe(T_jaw, 'dom_freq');
    out.jaw_bandratio_mean    = nanmean_safe(T_jaw, 'band_ratio_vib');

    % ---------- SPEECH summary ----------
    out.n_speech = height(T_speech);
    out.speech_std0_mean      = nanmean_safe(T_speech, 'std0');
    out.speech_range0_mean    = nanmean_safe(T_speech, 'range0');
    out.speech_bandratio_mean = nanmean_safe(T_speech, 'band_ratio_vib');
    out.speech_domfreq_mean   = nanmean_safe(T_speech, 'dom_freq');
    out.speech_repeat_corr    = compute_repeatability_from_rows(T_speech);

    % ---------- Subscores ----------
    % Static: lower std/range/drift is better
    out.static_noise_proxy = mean([...
        out.static_std0_mean,...
        out.static_range0_mean,...
        out.static_drift0_mean], 'omitnan');

    % Jaw: bigger dynamic range, autocorr peak, band ratio are better
    out.jaw_response_proxy = mean([...
        out.jaw_range0_mean,...
        out.jaw_autocorr_mean,...
        out.jaw_bandratio_mean], 'omitnan');

    % Speech: bigger band ratio + repeatability are better
    out.speech_info_proxy = mean([...
        out.speech_bandratio_mean,...
        out.speech_repeat_corr], 'omitnan');

    if isempty(pos_rows)
        pos_rows = out;
    else
        pos_rows(end+1) = out; %#ok<SAGROW>
    end
end

T_position_summary = struct2table(pos_rows);

%% ---------------- Normalize and score positions ----------------
T_position_summary.static_score = inverse_divmax(T_position_summary.static_noise_proxy);
T_position_summary.jaw_score    = direct_divmax(T_position_summary.jaw_response_proxy);
T_position_summary.speech_score = direct_divmax(T_position_summary.speech_info_proxy);
T_position_summary.total_score =...
    w_static * T_position_summary.static_score +...
    w_jaw    * T_position_summary.jaw_score +...
    w_speech * T_position_summary.speech_score;

pos_order = ["first","second","third","fourth","fifth","sixth","seventh","eighth","nineth"];

T_position_summary.position = categorical(T_position_summary.position, pos_order, 'Ordinal', true);
T_position_summary = sortrows(T_position_summary, 'position');
disp(' ');
disp('================ POSITION SUMMARY ================');
disp(T_position_summary);
plot_position_heatmap(T_position_summary, "total_score");

%% ---------------- Plot ranking ----------------
figure('Color','w');
bar(categorical(T_position_summary.position), T_position_summary.total_score, 0.7);
ylabel('Total quality score');
xlabel('Neck position');
title('Position ranking');
grid on;

%% ---------------- Optional save ----------------
writetable(T_file_features, 'T_file_features.csv');
writetable(T_position_summary, 'T_position_summary.csv');

fprintf('\nSaved:\n');
fprintf('  T_file_features.csv\n');
fprintf('  T_position_summary.csv\n');

%% ============================================================
%% FUNCTIONS
%% ============================================================

function meta = parse_map_filename(fname)
    s = lower(string(fname));
    s = erase(s, ".mat");

    meta = struct();
    meta.position = "unknown";
    meta.task = "unknown";
    meta.rep = NaN;

    % use "nineth" as canonical name to match your dataset naming
    pos_aliases = {
        "first",   ["first"];
        "second",  ["second"];
        "third",   ["third"];
        "fourth",  ["fourth"];
        "fifth",   ["fifth"];
        "sixth",   ["sixth"];
        "seventh", ["seventh"];
        "eighth",  ["eighth"];
        "nineth",  ["nineth","ninth"]
    };

    for i = 1:size(pos_aliases,1)
        canonical = pos_aliases{i,1};
        aliases = pos_aliases{i,2};
        for j = 1:numel(aliases)
            key = aliases(j);
            if contains(s, "_" + key + "_") || startsWith(s, "map_" + key + "_")
                meta.position = canonical;
                break;
            end
        end
        if meta.position ~= "unknown"
            break;
        end
    end

    task_names = ["static","jaw","speech"];
    for i = 1:numel(task_names)
        if contains(s, "_" + task_names(i) + "_")
            meta.task = task_names(i);
            break;
        end
    end

    tok = regexp(char(s), 'rep(\d+)', 'tokens', 'once');
    if ~isempty(tok)
        meta.rep = str2double(tok{1});
    end
end

function thr = choose_ref_threshold(ref, mode)
    ref = double(ref(:));
    switch lower(mode)
        case 'median'
            thr = median(ref, 'omitnan');
        case 'midrange'
            thr = (min(ref) + max(ref))/2;
        otherwise
            thr = median(ref, 'omitnan');
    end
end

function feat = extract_features_by_task(x, fs, task, band_low_1, band_low_2, band_vib, band_wide)
    x = double(x(:));
    x0 = x - mean(x, 'omitnan');

    feat = struct();

    % Basic robust features
    feat.mean_val = mean(x, 'omitnan');
    feat.std_val  = std(x, 'omitnan');
    feat.var_val  = var(x, 'omitnan');
    feat.min_val  = min(x);
    feat.max_val  = max(x);
    feat.range_val = max(x) - min(x);

    feat.mean0 = mean(x0, 'omitnan');
    feat.std0  = std(x0, 'omitnan');
    feat.rms0  = sqrt(mean(x0.^2, 'omitnan'));
    feat.range0 = max(x0) - min(x0);
    feat.mad0   = mad(x0, 1);

    % Drift proxy from moving average
    win_drift = max(5, round(0.5 * fs)); % 0.5 s
    x_ma = movmean(x, win_drift);
    feat.drift0 = max(x_ma) - min(x_ma);

    % PSD
    [f, Pxx] = compute_welch_psd(x0, fs);

    % Band energies
    feat.E_0p5_5   = band_energy_from_psd(f, Pxx, band_low_1);
    feat.E_5_50    = band_energy_from_psd(f, Pxx, band_low_2);
    feat.E_20_500  = band_energy_from_psd(f, Pxx, band_wide);
    feat.E_70_300  = band_energy_from_psd(f, Pxx, band_vib);
    feat.E_total   = trapz(f, Pxx);

    if feat.E_total > 0
        feat.band_ratio_vib = feat.E_70_300 / feat.E_total;
        feat.band_ratio_low = feat.E_0p5_5 / feat.E_total;
    else
        feat.band_ratio_vib = NaN;
        feat.band_ratio_low = NaN;
    end

    % Dominant frequency
    [feat.dom_freq, feat.dom_power] = dominant_frequency(f, Pxx, [0.5 min(500, fs/2 - 1)]);

    % Autocorrelation-based periodicity
    [feat.autocorr_peak, feat.autocorr_lag_s] = periodicity_feature(x0, fs);

    % Optional task-specific emphasis
    switch string(task)
        case "static"
            % nothing extra needed beyond stability features
        case "jaw"
            % same features are already useful
        case "speech"
            % same features are already useful
        otherwise
            % keep generic
    end
end

function val = nanmean_safe(T, varname)
    if ismember(varname, T.Properties.VariableNames)
        x = T.(varname);
        val = mean(x, 'omitnan');
    else
        val = NaN;
    end
end

function corr_mean = compute_repeatability_from_rows(Ttask)
    % Mean pairwise correlation among same-task repeats, after coarse time normalization.
    corr_mean = NaN;

    if height(Ttask) < 2
        return;
    end

    % Need original files to reload the actual analysis signals
    X = {};
    for i = 1:height(Ttask)
        fname = Ttask.file(i);
        if ~isfile(fname)
            continue;
        end
        S = load(fname);
        if ~isfield(S,'sig')
            continue;
        end

        sig = double(S.sig(:));
        fs_raw = infer_fs_from_loaded(sig, S);

        if isfield(S,'ref')
            ref = double(S.ref(:));
            try
                thr_ref = median(ref, 'omitnan');
                [x, ~, ~] = demodulate_phase_trimmed_mean(ref, sig, fs_raw, thr_ref);
            catch
                continue;
            end
        else
            x = sig;
        end

        x = double(x(:));
        x = x - mean(x, 'omitnan');

        if numel(x) < 20
            continue;
        end

        % Normalize amplitude
        sx = std(x, 'omitnan');
        if sx > 0
            x = x / sx;
        end

        % Resample to fixed length for simple correlation
        Ntarget = 3000;
        x_rs = resample_to_length(x, Ntarget);
        X{end+1} = x_rs; %#ok<AGROW>
    end

    if numel(X) < 2
        return;
    end

    C = [];
    for i = 1:numel(X)
        for j = i+1:numel(X)
            r = corrcoef(X{i}, X{j});
            if numel(r) >= 4
                C(end+1) = r(1,2); %#ok<AGROW>
            end
        end
    end

    if ~isempty(C)
        corr_mean = mean(C, 'omitnan');
    end
end

function y = resample_to_length(x, Ntarget)
    x = double(x(:));
    N = numel(x);
    if N == Ntarget
        y = x;
        return;
    end
    xi = linspace(1, N, Ntarget);
    y = interp1(1:N, x, xi, 'linear', 'extrap')';
end

function s = getfield_safe(S, fieldname, default_value)
    if isstruct(S) && isfield(S, fieldname)
        s = S.(fieldname);
    else
        s = default_value;
    end
end

function y = invert_minmax(x)
    x = double(x(:));
    good = ~isnan(x);
    y = nan(size(x));
    if sum(good) < 2
        y(good) = 1;
        return;
    end
    xmin = min(x(good));
    xmax = max(x(good));
    if xmax == xmin
        y(good) = 1;
    else
        y(good) = 1 - (x(good) - xmin) / (xmax - xmin);
    end
end

function y = direct_minmax(x)
    x = double(x(:));
    good = ~isnan(x);
    y = nan(size(x));
    if sum(good) < 2
        y(good) = 1;
        return;
    end
    xmin = min(x(good));
    xmax = max(x(good));
    if xmax == xmin
        y(good) = 1;
    else
        y(good) = (x(good) - xmin) / (xmax - xmin);
    end
end

function E = band_energy_from_psd(f, Pxx, band)
    if isempty(f) || isempty(Pxx) || numel(f) ~= numel(Pxx)
        E = NaN;
        return;
    end
    idx = (f >= band(1)) & (f <= band(2));
    if ~any(idx)
        E = 0;
        return;
    end
    E = trapz(f(idx), Pxx(idx));
end

function [df, dp] = dominant_frequency(f, Pxx, band)
    idx = (f >= band(1)) & (f <= band(2));
    if ~any(idx)
        df = NaN;
        dp = NaN;
        return;
    end
    ff = f(idx);
    pp = Pxx(idx);
    [dp, imax] = max(pp);
    df = ff(imax);
end

function [peak_val, lag_s] = periodicity_feature(x, fs)
    x = double(x(:));
    x = x - mean(x, 'omitnan');

    if numel(x) < round(0.5*fs)
        peak_val = NaN;
        lag_s = NaN;
        return;
    end

    [ac, lags] = xcorr(x, 'coeff');
    keep = lags >= 0;
    ac = ac(keep);
    lags = lags(keep);

    % Ignore lag=0 and search a reasonable region
    lag_min = max(1, round(0.05 * fs));  % 50 ms
    lag_max = min(numel(ac), round(1.0 * fs)); % 1 s

    if lag_max <= lag_min
        peak_val = NaN;
        lag_s = NaN;
        return;
    end

    seg = ac(lag_min:lag_max);
    [peak_val, idx] = max(seg);
    lag_s = (lag_min + idx - 1) / fs;
end

function [demod_sig, t_demod, out] = demodulate_phase_trimmed_mean(ref, sig, fs, thr_ref)
    % Cycle-wise demodulation with edge trimming.

    ref = double(ref(:));
    sig = double(sig(:));

    if length(ref) ~= length(sig)
        error('ref and sig must have the same length.');
    end

    ref_bin = ref > thr_ref;

    dref = diff(ref_bin);
    rise_idx = find(dref == 1) + 1;

    if numel(rise_idx) < 2
        error('Not enough rising edges detected for demodulation.');
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

        on_core_idx = trim_phase_indices(on_idx);
        off_core_idx = trim_phase_indices(off_idx);

        if isempty(on_core_idx) || isempty(off_core_idx)
            continue;
        end

        on_vals = sig_cycle(on_core_idx);
        off_vals = sig_cycle(off_core_idx);

        demod_sig(k) = mean(on_vals) - mean(off_vals);
        t_demod(k) = mean(idx_cycle) / fs;
        samples_per_cycle(k) = numel(idx_cycle);
        on_counts(k) = numel(on_vals);
        off_counts(k) = numel(off_vals);
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

    out.ref_bin = ref_bin;
    out.rise_idx = rise_idx;
    out.samples_per_cycle = samples_per_cycle;
    out.on_counts = on_counts;
    out.off_counts = off_counts;
    out.n_cycles = numel(demod_sig);
    out.fs_demod = fs_demod;
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
        core_idx = idx; % fallback
    end
end

function [f, Pxx] = compute_welch_psd(x, fs)
    x = double(x(:));
    x = x - mean(x, 'omitnan');

    if length(x) < 32 || isnan(fs) || fs <= 0
        f = NaN;
        Pxx = NaN;
        return;
    end

    if length(x) < 256
        wlen = max(32, floor(length(x)/4));
    else
        wlen = min(2048, floor(length(x)/4));
    end

    wlen = max(wlen, 16);
    win = hamming(wlen);
    noverlap = floor(0.5 * wlen);
    nfft = max(512, 2^nextpow2(wlen));

    [Pxx, f] = pwelch(x, win, noverlap, nfft, fs);
end

function fs = infer_fs_from_loaded(x, S)
    if nargin < 2
        S = struct();
    end

    if isfield(S, 'fs_est') && ~isempty(S.fs_est)
        fs = S.fs_est;
        return;
    end

    if isfield(S, 'fs') && ~isempty(S.fs)
        fs = S.fs;
        return;
    end

    if isfield(S, 'meta')
        if isfield(S.meta, 'frame_rate') && ~isempty(S.meta.frame_rate)
            fs = S.meta.frame_rate;
            return;
        end
        if isfield(S.meta, 'nominal_frame_rate') && ~isempty(S.meta.nominal_frame_rate)
            fs = S.meta.nominal_frame_rate;
            return;
        end
    end

    if isfield(S, 'duration_s') && ~isempty(S.duration_s)
        fs = numel(x) / S.duration_s;
        return;
    end

    if isfield(S, 'meta') && isfield(S.meta, 'duration_s') && ~isempty(S.meta.duration_s)
        fs = numel(x) / S.meta.duration_s;
        return;
    end

    error('Could not infer fs. Provide fs_est, fs, frame_rate, or duration_s.');
end

function plot_position_heatmap(T_position_summary, value_col)
    % Plot 3x3 heatmap using collection layout:
    %
    % 3 4 2
    % 5 1 6
    % 7 8 9
    %
    % Here 9 is named "nineth" to match dataset naming.

    if ischar(value_col)
        value_col = string(value_col);
    end

    if ~ismember("position", string(T_position_summary.Properties.VariableNames))
        error('T_position_summary must contain column "position".');
    end

    if ~ismember(value_col, string(T_position_summary.Properties.VariableNames))
        error('Column "%s" not found in T_position_summary.', value_col);
    end

    % custom layout from your experiment
    pos_grid = [...
        "third",  "fourth", "second";
        "fifth",  "first",  "sixth";
        "seventh","eighth", "nineth"
    ];

    M = nan(3,3);

    for r = 1:3
        for c = 1:3
            pos = pos_grid(r,c);
            idx = find(T_position_summary.position == pos, 1);

            if ~isempty(idx)
                M(r,c) = T_position_summary.(value_col)(idx);
            else
                M(r,c) = NaN;
            end
        end
    end

    figure('Color','w');
    imagesc(M);
    axis image;
    colormap(gray);
    colorbar;
    set(gca, 'XTick', 1:3, 'YTick', 1:3);
    set(gca, 'XTickLabel', {'col1','col2','col3'});
    set(gca, 'YTickLabel', {'row1','row2','row3'});
    title(sprintf('3x3 heatmap: %s', value_col), 'Interpreter', 'none');

    vmin = min(M(:), [], 'omitnan');
    vmax = max(M(:), [], 'omitnan');

    for r = 1:3
        for c = 1:3
            pos = pos_grid(r,c);
            val = M(r,c);

            if isnan(val)
                txt = sprintf('%s\nNaN', pos);
                txt_color = 'k';
            else
                % show both position name and your numeric label
                num_label = position_name_to_number(pos);
                txt = sprintf('%d\n%s\n%.3f', num_label, pos, val);

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
                'FontSize', 10);
        end
    end
end

function n = position_name_to_number(pos)
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
function y = direct_divmax(x)
    x = double(x(:));
    y = nan(size(x));

    good = ~isnan(x);
    if ~any(good)
        return;
    end

    xmax = max(x(good));
    if xmax == 0
        y(good) = NaN;
    else
        y(good) = x(good) / xmax;
    end
end

function y = inverse_divmax(x)
    % for metrics where smaller is better
    % score = min(x) / x
    x = double(x(:));
    y = nan(size(x));

    good = ~isnan(x) & x > 0;
    if ~any(good)
        return;
    end

    xmin = min(x(good));
    y(good) = xmin./ x(good);
end