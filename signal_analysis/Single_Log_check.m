clear; clc; close all;

load("first check_dark_led_off_rep1.mat");   % change filename
%%
sig = double(sig(:));
fs = numel(sig) / meta.duration_s;
t = (0:numel(sig)-1)'/fs;

sig0 = sig - median(sig);
sig_smooth = movmean(sig0, round(0.05*fs));   % 50 ms smoothing

figure;
plot(t, sig0, 'Color', [0.7 0.7 0.7]); hold on;
plot(t, sig_smooth, 'r', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Signal');
title(sprintf('%s | %s | rep%d', meta.mode, meta.condition, meta.repeat));
legend('raw centered', 'smoothed');