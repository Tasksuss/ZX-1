%% Cycle-wise demod from dual_channel_log.mat
clear; clc; close all;

load("dual_channel_log.mat");   % expects ref, sig, duration_s

ref = double(ref(:));
sig = double(sig(:));
N = numel(ref);
fs = N / duration_s;
t = (0:N-1)' / fs;

fprintf('Raw samples: %d\n', N);
fprintf('Estimated fs: %.2f Hz\n', fs);

%% 1) Hard threshold on ref
thr = 300;
ref_bin = ref > thr;

% crossings
d = diff(ref_bin);
rise_idx = find(d == 1);   % crossing between n and n+1
fall_idx = find(d == -1);

fprintf('Rising crossings: %d\n', numel(rise_idx));
fprintf('Falling crossings: %d\n', numel(fall_idx));

%% 2) Use rising-to-rising as one cycle
% Each cycle is [rise(k), rise(k+1)-1]
if numel(rise_idx) < 2
    error('Not enough rising edges detected.');
end

nCycles = numel(rise_idx) - 1;
demod_cycle = zeros(nCycles,1);
cycle_t = zeros(nCycles,1);
samples_per_cycle = zeros(nCycles,1);

for k = 1:nCycles
    i1 = rise_idx(k) + 1;
    i2 = rise_idx(k+1);

    if i2 <= i1 + 2
        demod_cycle(k) = NaN;
        cycle_t(k) = (i1 + i2)/(2*fs);
        samples_per_cycle(k) = i2 - i1 + 1;
        continue;
    end

    idx_cycle = i1:i2;
    ref_cycle = ref_bin(idx_cycle);
    sig_cycle = sig(idx_cycle);

    high_vals = sig_cycle(ref_cycle == 1);
    low_vals  = sig_cycle(ref_cycle == 0);

    if isempty(high_vals) || isempty(low_vals)
        demod_cycle(k) = NaN;
    else
        demod_cycle(k) = mean(high_vals) - mean(low_vals);
    end

    cycle_t(k) = mean(idx_cycle) / fs;
    samples_per_cycle(k) = numel(idx_cycle);
end

% remove invalid cycles
valid = ~isnan(demod_cycle);
demod_cycle = demod_cycle(valid);
cycle_t = cycle_t(valid);
samples_per_cycle = samples_per_cycle(valid);

fprintf('Valid cycles: %d\n', numel(demod_cycle));
fprintf('Mean samples/cycle: %.2f\n', mean(samples_per_cycle));
fprintf('Estimated cycle rate: %.2f Hz\n', numel(demod_cycle) / duration_s);

%% 3) Light smoothing just to inspect trend
% small moving average over 5 cycles
demod_smooth = movmean(demod_cycle, 5);

%% 4) Plot raw ref snippet
Nshow = min(round(0.05*fs), N);   % first 50 ms
figure;
plot(t(1:Nshow), ref(1:Nshow), 'b'); hold on;
yline(thr, '--k', 'thr=300');
xlabel('Time (s)');
ylabel('ref');
title('Raw ref with threshold');

%% 5) Plot demod signal
figure;
plot(cycle_t, abs(demod_cycle), '.-'); hold on;
plot(cycle_t, abs(demod_smooth), 'r', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('mean(high) - mean(low)');
title('Cycle-wise demodulated signal');
legend('per-cycle demod', 'smoothed');

%% 6) Optional quick summary
fprintf('Demod mean = %.3f\n', mean(demod_cycle));
fprintf('Demod std  = %.3f\n', std(demod_cycle));

high_all = sig(ref_bin == 1);
low_all  = sig(ref_bin == 0);

fprintf('Overall mean(high) = %.3f\n', mean(high_all));
fprintf('Overall mean(low)  = %.3f\n', mean(low_all));
fprintf('Overall difference = %.3f\n', mean(high_all) - mean(low_all));