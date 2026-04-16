clear; clc; close all;

%% ============================================================
%% STATIC BREATHING ICA SCAFFOLD (DEMOD FIRST)
%% ============================================================

%% ---------------- SETTINGS ----------------
region_keywords = {'first','second','third', ...
                   'fourth','fifth','sixth', ...
                   'seventh','eighth','nineth'};

thr_ref = 140;                  % hard threshold for reference
do_align_within_region = true;  % align repeats before averaging
max_lag_frac = 0.10;            % xcorr lag limit as fraction of length
min_repeats_required = 1;       % can set to 3 if you want stricter requirement
make_zscore = true;

apply_bandpass = true;          % NEW
bp_low = 0.05;                  % Hz, breathing lower edge
bp_high = 0.80;                 % Hz, breathing upper edge
bp_order = 2;                   % NEW

%% ---------------- FIND FILES ----------------
all_files = dir('*.mat');
all_names = {all_files.name};

% Only static files
keep_static = contains(lower(all_names), 'static');
static_files = all_files(keep_static);

if isempty(static_files)
    error('No *static*.mat files found in current folder.');
end

%% ---------------- SUMMARY ----------------
summary = summarise_static_files(static_files, region_keywords);

fprintf('\n=============================================\n');
fprintf('STATIC FILE SUMMARY\n');
fprintf('=============================================\n');
for i = 1:numel(summary)
    fprintf('%s : %d file(s)\n', summary(i).region, summary(i).count);
    for j = 1:numel(summary(i).files)
        fprintf('   %s\n', summary(i).files{j});
    end
end

%% ---------------- PREPARE MATRIX ----------------
[X, region_labels, prep_info, fs_common] = prepare_static_matrix( ...
    static_files, region_keywords, thr_ref, ...
    do_align_within_region, max_lag_frac, min_repeats_required, ...
    apply_bandpass, bp_low, bp_high, bp_order);

% X is regions x time
if make_zscore
    X_proc = zscore_rows(X);
else
    X_proc = X;
end

fprintf('\n=============================================\n');
fprintf('PREPARED MATRIX\n');
fprintf('=============================================\n');
fprintf('Number of regions included: %d\n', size(X_proc,1));
fprintf('Common length T: %d samples\n', size(X_proc,2));
fprintf('Common fs after demod/resampling: %.3f Hz\n', fs_common);

%% ---------------- ICA ----------------
ica2 = run_ica_quick(X_proc, 2);
ica3 = run_ica_quick(X_proc, 3);

%% ---------------- PLOTTING ----------------
t_norm = linspace(0, 1, size(X_proc,2));

% Region representative traces
figure('Name','Static region traces','Color','w');
tiledlayout(size(X_proc,1),1, 'Padding','compact', 'TileSpacing','compact');

for i = 1:size(X_proc,1)
    nexttile;
    plot(t_norm, X_proc(i,:), 'LineWidth', 1.0);
    ylabel(region_labels{i}, 'Interpreter','none');
    if i == 1
        title('Static breathing: demodulated representative traces');
    end
    if i == size(X_proc,1)
        xlabel('Normalised time');
    end
end

% ICA with 2 components
figure('Name','ICA 2 components','Color','w');
tiledlayout(2,2, 'Padding','compact', 'TileSpacing','compact');

nexttile;
plot(t_norm, ica2.sources(1,:), 'LineWidth', 1.0);
xlabel('Normalised time'); ylabel('IC1');
title('ICA (2 comps): source 1');

nexttile;
plot(t_norm, ica2.sources(2,:), 'LineWidth', 1.0);
xlabel('Normalised time'); ylabel('IC2');
title('ICA (2 comps): source 2');

nexttile([1 2]);
imagesc(ica2.loadings');
colorbar;
xlabel('Region index'); ylabel('Component');
title('ICA (2 comps): region loadings');
xticks(1:numel(region_labels));
xticklabels(region_labels);
xtickangle(45);

% ICA with 3 components
figure('Name','ICA 3 components','Color','w');
tiledlayout(3,2, 'Padding','compact', 'TileSpacing','compact');

nexttile;
plot(t_norm, ica3.sources(1,:), 'LineWidth', 1.0);
xlabel('Normalised time'); ylabel('IC1');
title('ICA (3 comps): source 1');

nexttile;
plot(t_norm, ica3.sources(2,:), 'LineWidth', 1.0);
xlabel('Normalised time'); ylabel('IC2');
title('ICA (3 comps): source 2');

nexttile;
plot(t_norm, ica3.sources(3,:), 'LineWidth', 1.0);
xlabel('Normalised time'); ylabel('IC3');
title('ICA (3 comps): source 3');

nexttile([2 1]);
imagesc(ica3.loadings');
colorbar;
xlabel('Region index'); ylabel('Component');
title('ICA (3 comps): region loadings');
xticks(1:numel(region_labels));
xticklabels(region_labels);
xtickangle(45);

%% PSD of recovered ICs
figure('Name','ICA source Welch PSD','Color','w');
tiledlayout(1,2, 'Padding','compact', 'TileSpacing','compact');

for k = 1:size(ica2.sources,1)
    [f, Pxx] = quick_welch_hz(ica2.sources(k,:), fs_common);
    nexttile;
    plot(f, Pxx, 'LineWidth', 1.0);
    title(sprintf('ICA2 - IC%d PSD', k));
    xlabel('Frequency (Hz)'); ylabel('dB/Hz');
    xlim([0, 1.5]);   % breathing-focused
end

% for k = 1:size(ica3.sources,1)
%     [f, Pxx] = quick_welch_hz(ica3.sources(k,:), fs_common);
%     nexttile;
%     plot(f, Pxx, 'LineWidth', 1.0);
%     title(sprintf('ICA3 - IC%d PSD', k));
%     xlabel('Frequency (Hz)'); ylabel('dB/Hz');
%     xlim([0, 1.5]);
% end

%% ---------------- PRINT PREP INFO ----------------
fprintf('\n=============================================\n');
fprintf('REGION PREPARATION INFO\n');
fprintf('=============================================\n');
for i = 1:numel(prep_info)
    fprintf('%s : %d repeat(s), original lengths = [', prep_info(i).region, prep_info(i).n_repeats);
    fprintf('%d ', prep_info(i).orig_lengths);
    fprintf('], common target length = %d\n', prep_info(i).target_len);
end


%% ============================================================
%% ======================== FUNCTIONS ==========================
%% ============================================================

function summary = summarise_static_files(static_files, region_keywords)
    summary = struct('region', {}, 'count', {}, 'files', {});
    fnames = {static_files.name};

    for i = 1:numel(region_keywords)
        key = lower(region_keywords{i});
        mask = contains(lower(fnames), key);
        summary(i).region = region_keywords{i};
        summary(i).count = sum(mask);
        summary(i).files = fnames(mask);
    end
end


function [X, region_labels, prep_info, fs_common] = prepare_static_matrix( ...
    static_files, region_keywords, thr_ref, do_align, max_lag_frac, ...
    min_repeats_required, apply_bandpass, bp_low, bp_high, bp_order)

    fnames = {static_files.name};
    region_signals = cell(numel(region_keywords),1);
    region_fs = cell(numel(region_keywords),1);
    region_dur = cell(numel(region_keywords),1);

    prep_info = struct('region', {}, 'n_repeats', {}, ...
                       'orig_lengths', {}, 'orig_fs', {}, ...
                       'orig_durations', {}, 'target_len', {});

    % Pass 1: load + demodulate all traces by region
    for i = 1:numel(region_keywords)
        key = lower(region_keywords{i});
        mask = contains(lower(fnames), key);
        these_files = fnames(mask);

        traces = {};
        fs_list = [];
        dur_list = [];
        lengths = [];

        for j = 1:numel(these_files)
            [x, fs_x] = load_and_demod_signal( ...
                these_files{j}, thr_ref, ...
                apply_bandpass, bp_low, bp_high, bp_order);

            if ~isempty(x)
                traces{end+1} = x(:); %#ok<AGROW>
                fs_list(end+1) = fs_x; %#ok<AGROW>
                lengths(end+1) = numel(x); %#ok<AGROW>
                dur_list(end+1) = numel(x) / fs_x; %#ok<AGROW>
            end
        end

        region_signals{i} = traces;
        region_fs{i} = fs_list;
        region_dur{i} = dur_list;

        prep_info(i).region = region_keywords{i};
        prep_info(i).n_repeats = numel(traces);
        prep_info(i).orig_lengths = lengths;
        prep_info(i).orig_fs = fs_list;
        prep_info(i).orig_durations = dur_list;

        if isempty(lengths)
            prep_info(i).target_len = NaN;
        else
            prep_info(i).target_len = min(lengths);
        end
    end

    % Keep only regions with enough repeats
    keep = false(numel(region_keywords),1);
    for i = 1:numel(region_keywords)
        keep(i) = prep_info(i).n_repeats >= min_repeats_required;
    end

    if ~any(keep)
        error('No regions passed the minimum repeat requirement.');
    end

    region_labels = region_keywords(keep);
    prep_info = prep_info(keep);
    region_signals = region_signals(keep);
    region_fs = region_fs(keep);
    region_dur = region_dur(keep);

    % Global common duration and length
    all_lengths = [];
    all_durations = [];

    for i = 1:numel(region_signals)
        these = region_signals{i};
        these_fs = region_fs{i};
        for j = 1:numel(these)
            all_lengths(end+1) = numel(these{j}); %#ok<AGROW>
            all_durations(end+1) = numel(these{j}) / these_fs(j); %#ok<AGROW>
        end
    end

    T_global = min(all_lengths);
    dur_common = median(all_durations);
    fs_common = T_global / dur_common;

    % Unify each region
    X = nan(numel(region_signals), T_global);

    for i = 1:numel(region_signals)
        traces = region_signals{i};
        avg_trace = unify_region_recordings(traces, T_global, do_align, max_lag_frac);
        X(i,:) = avg_trace(:).';
        prep_info(i).target_len = T_global;
    end
end

function avg_trace = unify_region_recordings(traces, target_len, do_align, max_lag_frac)
    % Resample all repeats to same length, optionally align to template,
    % then average across repeats.

    n = numel(traces);
    X = nan(n, target_len);

    for i = 1:n
        X(i,:) = resample_to_length(traces{i}, target_len);
    end

    if do_align && n > 1
        template = X(1,:);
        maxlag = max(1, round(max_lag_frac * target_len));

        for i = 2:n
            lag = estimate_lag_xcorr(X(i,:), template, maxlag);
            X(i,:) = shift_with_nan(X(i,:), lag);
        end
    end

    avg_trace = mean(X, 1, 'omitnan');

    nanmask = isnan(avg_trace);
    if any(nanmask)
        avg_trace = fillmissing(avg_trace, 'linear', 'EndValues', 'nearest');
    end
end


function [x, fs_x] = load_and_demod_signal(filename, thr_ref, apply_bandpass, bp_low, bp_high, bp_order)
    % Preferred order:
    %   1) demod_sig already saved
    %   2) raw ref + sig -> demodulate using trimmed phase-mean strategy
    %   3) fallback to sig if nothing else exists

    S = load(filename);

    if isfield(S, 'demod_sig')
        x = double(S.demod_sig(:));

        if isfield(S, 'fs_demod')
            fs_x = S.fs_demod;
        elseif isfield(S, 'meta') && isfield(S.meta, 'fs_demod')
            fs_x = S.meta.fs_demod;
        else
            error('demod_sig found in %s, but fs_demod is missing.', filename);
        end

        if apply_bandpass
            x = bandpass_filter(x, fs_x, bp_low, bp_high, bp_order);
        end
        return;
    end

    if isfield(S, 'ref') && isfield(S, 'sig')
        fs_raw = infer_fs_from_loaded(S.sig, S);
        [x, ~, out] = demodulate_phase_trimmed_mean(S.ref, S.sig, fs_raw, thr_ref);
        fs_x = out.fs_demod;

        if apply_bandpass
            x = bandpass_filter(x, fs_x, bp_low, bp_high, bp_order);
        end
        return;
    end

    if isfield(S, 'sig')
        warning('Using raw sig directly for %s because ref is missing.', filename);
        x = double(S.sig(:));
        fs_x = infer_fs_from_loaded(S.sig, S);

        if apply_bandpass
            x = bandpass_filter(x, fs_x, bp_low, bp_high, bp_order);
        end
        return;
    end

    warning('No recognised signal variables found in %s. Skipping.', filename);
    x = [];
    fs_x = NaN;
end

function y = bandpass_filter(x, fs, f_low, f_high, order)
    x = double(x(:));

    if f_high >= fs/2
        error('Bandpass upper edge must be below Nyquist.');
    end

    [b, a] = butter(order, [f_low f_high] / (fs/2), 'bandpass');
    y = filtfilt(b, a, x);
end

function y = resample_to_length(x, L)
    x = double(x(:));
    if numel(x) == L
        y = x(:).';
        return;
    end

    t_old = linspace(0, 1, numel(x));
    t_new = linspace(0, 1, L);
    y = interp1(t_old, x, t_new, 'linear');
end


function lag = estimate_lag_xcorr(x, template, maxlag)
    x = x(:);
    template = template(:);

    x = x - mean(x, 'omitnan');
    template = template - mean(template, 'omitnan');

    [c, lags] = xcorr(x, template, maxlag, 'coeff');
    [~, idx] = max(c);
    lag = lags(idx);
end


function y = shift_with_nan(x, lag)
    % Positive lag means shift right
    x = x(:).';
    N = numel(x);
    y = nan(1, N);

    if lag > 0
        y((lag+1):end) = x(1:(end-lag));
    elseif lag < 0
        lag2 = abs(lag);
        y(1:(end-lag2)) = x((lag2+1):end);
    else
        y = x;
    end
end


function Xz = zscore_rows(X)
    Xz = X;
    for i = 1:size(X,1)
        mu = mean(X(i,:), 'omitnan');
        sd = std(X(i,:), 'omitnan');
        if sd < eps
            sd = 1;
        end
        Xz(i,:) = (X(i,:) - mu) / sd;
    end
end


function out = run_ica_quick(X, nComp)
    % X is regions x time
    % ICA is run on observations x variables = time x regions

    Xobs = X.';   % T x R

    if exist('rica', 'file') == 2
        mdl = rica(Xobs, nComp, 'Standardize', false);
        S = transform(mdl, Xobs);        % T x nComp
        W = mdl.TransformWeights;        % nComp x R
        A = pinv(W);                     % approx loadings, R x nComp

        out.method = 'rica';
        out.sources = S.';               % nComp x T
        out.loadings = A;                % R x nComp
        out.W = W;
    elseif exist('fastica', 'file') == 2
        [icasig, A, W] = fastica(X, 'numOfIC', nComp, 'verbose', 'off');
        out.method = 'fastica';
        out.sources = icasig;            % nComp x T
        out.loadings = A;                % R x nComp
        out.W = W;
    else
        error('Neither rica nor fastica is available.');
    end
end


function [demod_sig, t_demod, out] = demodulate_phase_trimmed_mean(ref, sig, fs, thr_ref)
    % Cycle-wise demodulation with edge trimming.
    % For each rising-to-rising cycle:
    %   1) split into ON and OFF phase samples using thresholded ref
    %   2) discard edge samples within each phase block
    %   3) mean(ON_core) - mean(OFF_core)

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

        on_core_idx  = trim_phase_indices(on_idx);
        off_core_idx = trim_phase_indices(off_idx);

        if isempty(on_core_idx) || isempty(off_core_idx)
            continue;
        end

        on_vals  = sig_cycle(on_core_idx);
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
        core_idx = idx;
    end
end


function fs = infer_fs_from_loaded(x, S)
    % Preference order:
    %   1) fs_est
    %   2) fs
    %   3) meta.frame_rate
    %   4) meta.nominal_frame_rate
    %   5) duration_s
    %   6) meta.duration_s

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

function [f, Pxx] = quick_welch_hz(x, fs)
    x = double(x(:));
    x = x - mean(x, 'omitnan');

    wlen = 1024*8;
    w = hamming(wlen);
    noverlap = floor(0.5 * wlen);

    % Base FFT length from window length
    nfft_base = max(512, 2^nextpow2(wlen));

    % x9 zero padding for denser frequency grid
    nfft = 9 * nfft_base;

    [Pxx, f] = pwelch(x, w, noverlap, nfft, fs);
end