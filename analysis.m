clear; clc; close all;

%% ============================================================
% SECTION 1 — 850 nm IR, DC recording: single trial overview
% Expected variables in file:
%   sig
% Optional:
%   duration_s, meta.duration_s, fs, fs_est, meta.frame_rate
% ============================================================

filename = "DC_arm_press_rep4";

if strlength(filename) > 0
    load(filename);

    sig = double(sig(:));
    fs = fs_est;

    stats_dc = compute_first_order_stats(sig);
    [bin_centres_dc, pdf_dc] = compute_pde_hist(sig, 60);
    [f_dc, Pxx_dc] = compute_welch_psd(sig, fs);
    sig_ma = moving_average_filter(sig, round(0.05 * fs));   % 50 ms

    t = (0:length(sig)-1)'/fs;

    figure;
    % subplot(3,1,1);
    plot(t, sig); hold on;
    plot(t, sig_ma, 'LineWidth', 1.2);
    xlabel('Time (s)'); ylabel('ADC');
    ylim([500 550]);
    title('850 nm DC: raw signal + moving average');
    legend('raw','moving avg');
    % 
    % subplot(3,1,2);
    % bar(bin_centres_dc, pdf_dc, 1);
    % xlabel('ADC value'); ylabel('PDF');
    % title('850 nm DC: histogram-based PDE');
    % 
    % subplot(3,1,3);
    % plot(f_dc, 10*log10(Pxx_dc + eps), 'LineWidth', 1.1);
    % xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
    % title('850 nm DC: Welch PSD');
    % 
    % disp('SECTION 1 — DC single-trial stats');
    % disp(stats_dc);
end


%% ============================================================
% SECTION 2 — 850 nm IR, DC recording: two-condition comparison
% Example use:
%   LED off vs LED on
%   arm no motion vs arm pressing
%   neck midline vs neck lateral
% Expected variables in each file:
%   sig
% ============================================================

file_A = "DC_dark_led_off_rep2";
file_B = "DC_ambient_led_on_rep2";

if strlength(file_A) > 0 && strlength(file_B) > 0
    S1 = load(file_A);
    S2 = load(file_B);

    x1 = double(S1.sig(:));
    x2 = double(S2.sig(:));

    [bc1, p1] = compute_pde_hist(x1, 60);
    [bc2, p2] = compute_pde_hist(x2, 60);

    stats_A = compute_first_order_stats(x1);
    stats_B = compute_first_order_stats(x2);

    figure;
    % subplot(2,1,1);
    % plot(x1, 'LineWidth', 1.0); hold on;
    % plot(x2, 'LineWidth', 1.0);
    % xlabel('Sample'); ylabel('ADC');
    % title('DC comparison: raw traces');
    % legend('A','B');

    % subplot(2,1,2);
    bar(bc1, p1); hold on;
    bar(bc2, p2);
    xlabel('ADC value'); ylabel('PDF');
    title('DC comparison: histogram-based PDE');
    legend('A','B');

    disp('SECTION 2 — DC comparison stats');
    disp(table(stats_A.mean_val, stats_A.std_val, stats_A.var_val, stats_A.min_val, stats_A.max_val, ...
               stats_B.mean_val, stats_B.std_val, stats_B.var_val, stats_B.min_val, stats_B.max_val, ...
               'VariableNames', {'A_mean','A_std','A_var','A_min','A_max', ...
                                 'B_mean','B_std','B_var','B_min','B_max'}));
end


%% ============================================================
% SECTION 3 — 850 nm IR, AC / AM recording: demodulation
% Expected variables in file:
%   ref, sig
% Optional:
%   duration_s, meta.duration_s, fs, fs_est, meta.nominal_frame_rate
%
% Demodulation strategy:
%   threshold ref at thr_ref
%   find rising-edge cycles
%   for each cycle: mean(sig | ref=1) - mean(sig | ref=0)
% ============================================================

filename = "map_eighth_speech_rep3";
thr_ref = 140;

if strlength(filename) > 0
    load(filename);

    ref = double(ref(:));
    sig = double(sig(:));
    fs = length(sig)/duration_s;

    [demod_sig, t_demod, demod_info] = demodulate_phase_trimmed_mean(ref, sig, fs, thr_ref);

    stats_demod = compute_first_order_stats(demod_sig);
    [bc_demod, p_demod] = compute_pde_hist(demod_sig, 60);
    [f_demod, Pxx_demod] = compute_welch_psd(demod_sig, demod_info.fs_demod);

    t_raw = (0:length(sig)-1)'/fs;
    Nshow = min(round(0.1*fs), length(sig));   % first 100 ms

    figure;
    subplot(2,1,1);
    plot(t_raw(1:Nshow), ref(1:Nshow), 'LineWidth', 1.0); hold on;
    yline(thr_ref, '--k', 'thr');
    xlabel('Time (s)'); ylabel('ADC');
    title('AC: reference trace with threshold');
    
    % subplot(4,1,2);
    % plot(t_raw(1:Nshow), sig(1:Nshow), 'LineWidth', 1.0);
    % xlabel('Time (s)'); ylabel('ADC');
    % title('AC: raw sensor signal');

    subplot(2,1,2);
    plot(t_demod, demod_sig, '.-'); hold on;
    plot(t_demod, moving_average_filter(demod_sig, 5), 'LineWidth', 1.2);
    xlabel('Time (s)'); ylabel('Demod amplitude');
    title('AC: demodulated envelope');
    legend('per-cycle demod','5-point moving avg');

    % subplot(4,1,4);
    % plot(f_demod, 10*log10(Pxx_demod + eps), 'LineWidth', 1.1);
    % xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
    % title('AC: Welch PSD of demodulated signal');
    % 
    % figure;
    % bar(bc_demod, p_demod, 1);
    % xlabel('Demod amplitude'); ylabel('PDF');
    % title('AC: histogram-based PDE of demodulated signal');
    % 
    % disp('SECTION 3 — AC demod stats');
    % disp(stats_demod);
    % fprintf('Demod fs ≈ %.3f Hz\n', demod_info.fs_demod);
    % fprintf('Detected %d valid cycles\n', demod_info.n_cycles);
end


%% ============================================================
% SECTION 4 — 850 nm IR, AC / AM recording: compare two trials
% Example use:
%   ambient LED on vs static arm
%   neck no motion vs jaw movement
%   lateral vs midline for same task
% Expected variables in each file:
%   ref, sig
% ============================================================

file_A = "first_ambient_led_on_rep1";
file_B = "first_arm_static_rep1";
thr_ref = 150;

if strlength(file_A) > 0 && strlength(file_B) > 0
    S1 = load(file_A);
    S2 = load(file_B);

    ref1 = double(S1.ref(:));
    sig1 = double(S1.sig(:));
    fs1 = infer_fs_from_loaded(sig1, S1);

    ref2 = double(S2.ref(:));
    sig2 = double(S2.sig(:));
    fs2 = infer_fs_from_loaded(sig2, S2);

    [y1, t1, info1] = demodulate_phase_trimmed_mean(ref1, sig1, fs1, thr_ref);
    y1 = moving_average_filter(y1, 5);
    [y2, t2, info2] = demodulate_phase_trimmed_mean(ref2, sig2, fs2, thr_ref);
    y2 = moving_average_filter(y2, 5);

    [bc1, p1] = compute_pde_hist(y1, 60);
    [bc2, p2] = compute_pde_hist(y2, 60);

    figure;
    subplot(2,1,1);
    plot(t1, y1, 'LineWidth', 1.0); hold on;
    plot(t2, y2, 'LineWidth', 1.0);
    xlabel('Time (s)'); ylabel('Demod amplitude');
    title('AC comparison: demodulated traces');
    legend('Ambient light','On the skin');

    subplot(2,1,2);
    plot(bc1, p1, 'LineWidth', 1.2); hold on;
    plot(bc2, p2, 'LineWidth', 1.2);
    xlabel('Demod amplitude'); ylabel('PDF');
    title('AC comparison: histogram-based PDE');
    legend('Ambient light','On the skin');

    disp('SECTION 4 — AC comparison stats');
    disp(table(mean(y1), std(y1), mean(y2), std(y2), info1.fs_demod, info2.fs_demod, ...
        'VariableNames', {'A_mean','A_std','B_mean','B_std','A_fs_demod','B_fs_demod'}));
end


%% ============================================================
% SECTION 5 — Alternating RED / empty / IR / empty PPG validation
% Expected variables in file:
%   states, sig
% Optional:
%   meta.led_rate_hz
% State mapping:
%   0 = RED ON
%   1 = OFF after RED
%   2 = IR ON
%   3 = OFF after IR
% ============================================================

load("ppg_alt_red_ir5")
ppg_band = [0.5 3];    % Hz

t_elapse = 2;  % seconds to trim from both beginning and end

% Calculate sample indices for trimming
trim_start_samples = round(t_elapse * fs_led);
trim_end_samples = round(t_elapse * fs_led);

% Create the full time vector first (before trimming)
t_full = (0:length(red_corr)-1)' / fs_led;

% Trim the signals
red_corr_trimmed = red_corr(trim_start_samples+1 : end-trim_end_samples);
ir_corr_trimmed = ir_corr(trim_start_samples+1 : end-trim_end_samples);

% Apply bandpass filter to the FULL signals first
red_bp_full = bandpass_filter(red_corr, fs_led, ppg_band(1), ppg_band(2), 2);
ir_bp_full = bandpass_filter(ir_corr, fs_led, ppg_band(1), ppg_band(2), 2);

% Then trim the bandpassed signals
red_bp = red_bp_full(trim_start_samples+1 : end-trim_end_samples);
ir_bp = ir_bp_full(trim_start_samples+1 : end-trim_end_samples);

% Create the correct time vector for the trimmed signals
t_trimmed = t_full(trim_start_samples+1 : end-trim_end_samples);

% Now t_trimmed matches red_bp and ir_bp in length

% For PSD calculation, you can use either the full or trimmed signals
% Here I'll use the trimmed corrected signals
[f_red, P_red] = compute_welch_psd(red_corr_trimmed, fs_led);
[f_ir,  P_ir ] = compute_welch_psd(ir_corr_trimmed, fs_led);

stats_red = compute_first_order_stats(red_corr_trimmed);
stats_ir  = compute_first_order_stats(ir_corr_trimmed);

% Plotting with the correct time vector
figure;
subplot(2,1,1);
plot(t_trimmed, red_bp, 'LineWidth', 1.1); hold on;
plot(t_trimmed, ir_bp,  'LineWidth', 1.1);
xlabel('Time (s)'); ylabel('Bandpassed');
title('Alternating red/IR: bandpassed PPG-like signals');
legend('red bp','IR bp');

subplot(2,1,2);
plot(f_red, 10*log10(P_red + eps), 'LineWidth', 1.1); hold on;
plot(f_ir,  10*log10(P_ir  + eps), 'LineWidth', 1.1);
xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
title('Alternating red/IR: Welch PSD');
legend('red corr','IR corr');
xlim([0 10]); 
    % subplot(3,1,1);
    % plot(t_led, red_corr, 'LineWidth', 1.0); hold on;
    % plot(t_led, ir_corr,  'LineWidth', 1.0);
    % xlabel('Time (s)'); ylabel('Corrected ADC');
    % title('Alternating red/IR: corrected channels');
    % legend('red corr','IR corr');

    disp('SECTION 5 — PPG branch stats');
    disp(table(stats_red.mean_val, stats_red.std_val, stats_ir.mean_val, stats_ir.std_val, ...
        'VariableNames', {'red_mean','red_std','ir_mean','ir_std'}));

%% ============================================================
% ======================== FUNCTIONS ==========================
% ============================================================

function [demod_sig, t_demod, out] = demodulate_phase_mean_difference(ref, sig, fs, thr_ref)
    % Demodulation by cycle-wise phase averaging:
    % one datapoint per carrier cycle:
    %   mean(sig | ref=1) - mean(sig | ref=0)
    %
    % Inputs:
    %   ref     - reference channel
    %   sig     - sensor channel
    %   fs      - raw sampling frequency
    %   thr_ref - hard threshold for ref
    %
    % Outputs:
    %   demod_sig - one value per detected cycle
    %   t_demod   - one timestamp per detected cycle
    %   out       - diagnostic struct

    ref = double(ref(:));
    sig = double(sig(:));

    if length(ref) ~= length(sig)
        error('ref and sig must have the same length.');
    end

    % Threshold the reference
    ref_bin = ref > thr_ref;

    % Rising edges: transition 0 -> 1
    dref = diff(ref_bin);
    rise_idx = find(dref == 1) + 1;   % first sample above threshold

    if numel(rise_idx) < 2
        error('Not enough rising edges detected for demodulation.');
    end

    % Use rising-to-rising intervals as cycles
    nCycles = numel(rise_idx) - 1;

    demod_sig = nan(nCycles,1);
    t_demod = nan(nCycles,1);
    samples_per_cycle = nan(nCycles,1);

    for k = 1:nCycles
        i1 = rise_idx(k);
        i2 = rise_idx(k+1) - 1;

        if i2 <= i1
            continue;
        end

        idx_cycle = i1:i2;
        ref_cycle = ref_bin(idx_cycle);
        sig_cycle = sig(idx_cycle);

        on_vals  = sig_cycle(ref_cycle == 1);
        off_vals = sig_cycle(ref_cycle == 0);

        if isempty(on_vals) || isempty(off_vals)
            continue;
        end

        demod_sig(k) = mean(on_vals) - mean(off_vals);
        t_demod(k) = mean(idx_cycle) / fs;
        samples_per_cycle(k) = numel(idx_cycle);
    end

    % Remove invalid cycles
    valid = ~isnan(demod_sig) & ~isnan(t_demod);
    demod_sig = demod_sig(valid);
    t_demod = t_demod(valid);
    samples_per_cycle = samples_per_cycle(valid);

    % Demodulated sample rate
    if numel(t_demod) >= 2
        fs_demod = 1 / median(diff(t_demod));
    else
        fs_demod = NaN;
    end

    out.ref_bin = ref_bin;
    out.rise_idx = rise_idx;
    out.samples_per_cycle = samples_per_cycle;
    out.n_cycles = numel(demod_sig);
    out.fs_demod = fs_demod;
end

function [demod_sig, t_demod, out] = demodulate_phase_trimmed_mean(ref, sig, fs, thr_ref)
    % Cycle-wise demodulation with edge trimming.
    % For each rising-to-rising cycle:
    %   1) split into ON and OFF phase samples using thresholded ref
    %   2) discard edge samples within each contiguous phase block
    %   3) compute mean(ON_core) - mean(OFF_core)
    %
    % This is designed for low samples-per-cycle cases (~8-10 samples/cycle),
    % where edge points are unreliable.

    ref = double(ref(:));
    sig = double(sig(:));

    if length(ref) ~= length(sig)
        error('ref and sig must have the same length.');
    end

    ref_bin = ref > thr_ref;

    % Rising edges define cycle boundaries
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

        % Extract contiguous ON and OFF runs inside this cycle
        on_idx = find(ref_cycle == 1);
        off_idx = find(ref_cycle == 0);

        if isempty(on_idx) || isempty(off_idx)
            continue;
        end

        % Keep only central samples of each phase
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
    % idx is a vector of positions within one cycle belonging to a phase.
    % We discard edge points to keep only the plateau / central points.
    %
    % Heuristic:
    %   length >= 5  -> drop first and last 2
    %   length == 4  -> keep middle 2
    %   length == 3  -> keep middle 1
    %   length <= 2  -> keep all (fallback)

    n = numel(idx);

    if n >= 5
        core_idx = idx(3:end-2);
    elseif n == 4
        core_idx = idx(2:3);
    elseif n == 3
        core_idx = idx(2);
    else
        core_idx = idx;   % fallback when too few points exist
    end
end

function [bin_centres, pdf_est] = compute_pde_hist(x, nbin)
    % Histogram-only PDE estimate, no kernel fitting.
    x = double(x(:));
    [counts, edges] = histcounts(x, nbin, 'Normalization', 'pdf');
    bin_centres = (edges(1:end-1) + edges(2:end))/2;
    pdf_est = counts(:);
end


function stats = compute_first_order_stats(x)
    x = double(x(:));
    stats.mean_val = mean(x);
    stats.std_val = std(x);
    stats.var_val = var(x);
    stats.min_val = min(x);
    stats.max_val = max(x);
    stats.range_val = max(x) - min(x);
    stats.median_val = median(x);
    stats.skewness_val = skewness(x);
    stats.kurtosis_val = kurtosis(x);
end


function [f, Pxx] = compute_welch_psd(x, fs)
    x = double(x(:));
    x = x - mean(x, 'omitnan');

    if length(x) < 256
        wlen = max(32, floor(length(x)/4));
    else
        wlen = 1024;
    end

    wlen = max(wlen, 16);
    win = hamming(wlen);
    noverlap = floor(0.5 * wlen);
    nfft = max(512, 2^nextpow2(wlen));

    [Pxx, f] = pwelch(x, win, noverlap, nfft, fs);
end


function y = bandpass_filter(x, fs, f_low, f_high, order)
    x = double(x(:));
    [b, a] = butter(order, [f_low f_high]/(fs/2), 'bandpass');
    y = filtfilt(b, a, x);
end


function y = moving_average_filter(x, win)
    x = double(x(:));
    y = movmean(x, win);
end

function fs = infer_fs_from_loaded(x, S)
    % Infer sampling frequency from loaded variables/metadata.
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