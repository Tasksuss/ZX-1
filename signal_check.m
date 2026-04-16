%% load data
clc
clear
close all

load('dual_channel_log.mat')
%% --- Hard-threshold crossing method for ref ---
thr = 300;

ref = double(ref(:));
sig = double(sig(:));
N   = numel(ref);
fs  = N / duration_s;
t   = (0:N-1)'/fs;

% Binary state from hard threshold
ref_bin = ref > thr;

% Crossing indices:
% +1 means rising crossing between n and n+1
% -1 means falling crossing between n and n+1
d = diff(ref_bin);
rise_idx = find(d == 1);
fall_idx = find(d == -1);
cross_idx = sort([rise_idx; fall_idx]);

% Optional: interpolate crossing times for sub-sample accuracy
cross_t = zeros(size(cross_idx));
for k = 1:numel(cross_idx)
    n = cross_idx(k);
    y1 = ref(n);
    y2 = ref(n+1);

    if y2 ~= y1
        alpha = (thr - y1) / (y2 - y1);
        alpha = max(0,min(1,alpha));
    else
        alpha = 0.5;
    end

    cross_t(k) = t(n) + alpha/fs;
end

% Reconstruct clean square reference by thresholding directly
ref_clean = 2*double(ref_bin) - 1;   % +/-1

% Estimate carrier frequency from successive same-type crossings
if numel(rise_idx) >= 2
    T_est = median(diff(rise_idx)) / fs;   % seconds
    f_est = 1 / T_est;
elseif numel(fall_idx) >= 2
    T_est = median(diff(fall_idx)) / fs;
    f_est = 1 / T_est;
else
    f_est = NaN;
end

fprintf('Detected %d rising and %d falling crossings\n', numel(rise_idx), numel(fall_idx));
fprintf('Estimated carrier frequency = %.2f Hz\n', f_est);

%% --- Demod using hard-threshold clean ref ---
sig0 = sig - mean(sig);
mix  = sig0 .* ref_clean;

lp_len = max(5, round(2*fs/f_est));   % ~2 carrier periods
demod = movmean(mix, lp_len);

%% --- Welch PSD ---
nwin = min(4096, 2^nextpow2(round(fs)));
nwin = max(nwin, 256);
noverlap = round(0.5*nwin);
nfft = max(4096, 2^nextpow2(nwin));

[Pref, fWelch] = pwelch(ref-mean(ref), hamming(nwin), noverlap, nfft, fs);
[Psig, ~]      = pwelch(sig-mean(sig), hamming(nwin), noverlap, nfft, fs);
[Pdem, fDem]   = pwelch(demod-mean(demod), hamming(nwin), noverlap, nfft, fs);

%% --- Plot snippet ---
Nshow = min(round(0.05*fs), N);  % 50 ms

figure;
plot(t(1:Nshow), ref(1:Nshow), 'b', 'LineWidth', 1); hold on;
yline(thr, '--k', 'thr=300');
stairs(t(1:Nshow), ...
    (ref_clean(1:Nshow)+1)/2 * (max(ref(1:Nshow))-min(ref(1:Nshow))) + min(ref(1:Nshow)), ...
    'r', 'LineWidth', 1.2);

rise_show = rise_idx(rise_idx <= Nshow-1);
fall_show = fall_idx(fall_idx <= Nshow-1);

plot(cross_t(ismember(cross_idx, rise_show)), thr*ones(sum(ismember(cross_idx, rise_show)),1), ...
    'go', 'MarkerSize', 5, 'LineWidth', 1.2);
plot(cross_t(ismember(cross_idx, fall_show)), thr*ones(sum(ismember(cross_idx, fall_show)),1), ...
    'mo', 'MarkerSize', 5, 'LineWidth', 1.2);

xlabel('Time (s)');
ylabel('Amplitude');
title('Raw ref with hard threshold crossings');
legend('raw ref','threshold','clean ref','rising','falling');

%% --- PSD plots ---
figure;
plot(fWelch, 10*log10(Pref+eps), 'LineWidth', 1); hold on;
plot(fWelch, 10*log10(Psig+eps), 'LineWidth', 1);
xline(50,'--k','50 Hz');
xline(500,'--r','500 Hz');
xlim([0 1500]);
xlabel('Frequency (Hz)');
ylabel('PSD / dB/Hz');
title('Welch PSD of raw ref and raw sig');
legend('ref raw','sig raw');

figure;
plot(fDem, 10*log10(Pdem+eps), 'LineWidth', 1);
xline(50,'--k','50 Hz');
xlim([0 100]);
xlabel('Frequency (Hz)');
ylabel('PSD / dB/Hz');
title('Welch PSD of demodulated baseband');