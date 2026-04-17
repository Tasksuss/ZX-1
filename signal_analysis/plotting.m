%% ========================================================================
%  SIMPLE PLOTTING SCRIPT FOR DC, AC, AND PPG DATA
%  ========================================================================
%  This script:
%    - Loops over all .mat files in a specified folder
%    - Applies mode‑specific preprocessing
%    - Plots the processed signal(s) with absolute time on the x‑axis
%    - Saves the figure as both .png and .fig in the same folder
%
%  Helper functions (must be on path):
%    infer_fs_from_loaded, moving_average_filter,
%    demodulate_phase_trimmed_mean, bandpass_filter
% ========================================================================

clear; clc; close all;

%% ======================= USER SETTINGS ==================================
mode = 'AC';                % 'DC', 'AC', or 'PPG'
data_folder = 'C:\Users\yangh\OneDrive\Desktop\Y3 Project\ZX-1_Map';  % <-- CHANGE THIS to your folder path

% Optional parameters (adjust as needed)
moving_avg_window_s = 0.1;     % DC moving average window (seconds)
ac_thr_ref = 140;               % Threshold for AC reference channel
ppg_band = [0.5 3];             % Bandpass limits for PPG (Hz)
trim_time_s = 2;                % Seconds to trim from start/end of PPG
% =========================================================================

%% ======================= PROCESS ALL .MAT FILES =========================
mat_files = dir(fullfile(data_folder, '*.mat'));

if isempty(mat_files)
    error('No .mat files found in: %s', data_folder);
end

for i = 1:length(mat_files)
    file_path = fullfile(mat_files(i).folder, mat_files(i).name);
    [~, file_name, ~] = fileparts(mat_files(i).name);
    
    fprintf('Processing %s (%d/%d)...\n', file_name, i, length(mat_files));
    
    % Load the data
    data = load(file_path);
    
    switch upper(mode)
        % ----------------------------------------------------------------
        case 'DC'
            % Expect variable 'sig'
            if ~isfield(data, 'sig')
                warning('File "%s" missing ''sig''. Skipping.', file_name);
                continue;
            end
            sig = double(data.sig(:));
            
            % Determine sampling frequency
            fs = infer_fs_from_loaded(sig, data);
            
            % Apply moving average filter
            sig_ma = moving_average_filter(sig, round(moving_avg_window_s * fs));
            
            % Time vector (absolute time)
            t = (0:length(sig)-1)' / fs;
            
            % Create plot
            figure('Name', file_name);
            plot(t, sig, 'Color', [0.5 0.5 0.5]); hold on;
            plot(t, sig_ma, 'LineWidth', 1.2);
            xlabel('Time (s)');
            ylabel('ADC');
            title(sprintf('DC Signal: %s', strrep(file_name, '_', '\_')));
            legend('Raw', 'Moving avg', 'Location', 'best');
            grid on;
            
        % ----------------------------------------------------------------
        case 'AC'
            % Expect variables 'ref' and 'sig'
            if ~all(isfield(data, {'ref', 'sig'}))
                warning('File "%s" missing ''ref'' or ''sig''. Skipping.', file_name);
                continue;
            end
            ref = double(data.ref(:));
            sig = double(data.sig(:));
            
            % Determine sampling frequency
            fs = infer_fs_from_loaded(sig, data);
            
            % Check if LED is always off (no rising edges above threshold)
            if all(ref <= ac_thr_ref)
                % Constant off: no demodulation, plot raw signal directly
                t_plot = (0:length(sig)-1)' / fs;
                sig_plot = sig;
                sig_smooth = moving_average_filter(sig, round(moving_avg_window_s * fs));
                y_label = 'ADC';
                title_prefix = 'AC Raw Signal (LED off)';
                legend_str = {'Raw', 'Moving avg'};
            else
                % Normal AC: demodulate using trimmed mean difference
                [demod_sig, t_demod, demod_info] = demodulate_phase_trimmed_mean(ref, sig, fs, ac_thr_ref);
                t_plot = t_demod;
                sig_plot = demod_sig;
                sig_smooth = moving_average_filter(demod_sig, 5);
                y_label = 'Demod amplitude';
                title_prefix = 'AC Demodulated Envelope';
                legend_str = {'Per-cycle demod (moving averaged)'};
            end
            
            % Create plot with uniform color style (grey raw, blue smooth)
            figure('Name', file_name);
            %plot(t_plot, sig_plot, '.-', 'Color', [0.5 0.5 0.5], 'MarkerSize', 4); hold on;
            plot(t_plot, sig_smooth, 'LineWidth', 1.2);
            xlabel('Time (s)');
            ylabel(y_label);
            title(sprintf('%s: %s', title_prefix, strrep(file_name, '_', '\_')));
            legend(legend_str, 'Location', 'best');
            grid on;
            
        % ----------------------------------------------------------------
                % ----------------------------------------------------------------
        case 'PPG'
            % Expect variables 'red_corr', 'ir_corr', and sampling info
            if ~all(isfield(data, {'red_corr', 'ir_corr'}))
                warning('File "%s" missing ''red_corr'' or ''ir_corr''. Skipping.', file_name);
                continue;
            end
            red_corr = double(data.red_corr(:));
            ir_corr  = double(data.ir_corr(:));
            
            % Correctly infer ADC sampling rate from meta data
            % (uses fs_est, fs, meta.frame_rate, meta.duration_s, etc.)
            try
                fs = infer_fs_from_loaded(red_corr, data);
            catch
                % Fallback: if meta contains 'adc_fs' or 'slot_rate'
                if isfield(data, 'meta') && isfield(data.meta, 'adc_fs')
                    fs = data.meta.adc_fs;
                elseif isfield(data, 'meta') && isfield(data.meta, 'slot_rate_hz')
                    fs = data.meta.slot_rate;
                else
                    error('Could not determine ADC sampling rate. Check meta fields.');
                end
            end
            
            % Bandpass filter the full signals
            red_bp_full = bandpass_filter(red_corr, fs, ppg_band(1), ppg_band(2), 2);
            ir_bp_full  = bandpass_filter(ir_corr,  fs, ppg_band(1), ppg_band(2), 2);
            
            % Trim edges to remove filter transients
            trim_samples = round(trim_time_s * fs);
            if length(red_corr) <= 2 * trim_samples
                warning('File "%s" too short for trimming. Skipping.', file_name);
                continue;
            end
            idx_keep = (trim_samples+1) : (length(red_corr)-trim_samples);
            red_bp = red_bp_full(idx_keep);
            ir_bp  = ir_bp_full(idx_keep);
            
            % Time vector using ADC sampling rate
            t_full = (0:length(red_corr)-1)' / fs;
            t_trim = t_full(idx_keep);
            
            % Create plot
            figure('Name', file_name);
            plot(t_trim, red_bp, 'LineWidth', 1.1); hold on;
            plot(t_trim, ir_bp, 'LineWidth', 1.1);
            xlabel('Time (s)');
            ylabel('Bandpassed amplitude');
            title(sprintf('PPG Red/IR Channels: %s', strrep(file_name, '_', '\_')));
            legend('Red', 'IR', 'Location', 'best');
            grid on;
    end
    
    % Save the figure as PNG and FIG in the same folder
    saveas(gcf, fullfile(data_folder, [file_name '.png']));
    %saveas(gcf, fullfile(data_folder, [file_name '.fig']));
    
    % Optionally close the figure to save memory (uncomment if desired)
    close(gcf);
end

fprintf('Done. All figures saved to: %s\n', data_folder);


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