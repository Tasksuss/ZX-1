% ============================================================
% Demodulation illustration (concept figure)
% Generates a report-style schematic figure for:
%   1) reference square wave
%   2) raw modulated sensor signal
%   3) ON/OFF phase means
%   4) demodulated envelope
%
% Save as:
%   figure_demodulation_illustration.m
% ============================================================

clear; clc; close all;

%% ---------------- Parameters ----------------
fs = 50000;          % plotting sample rate for smooth concept figure
T  = 0.08;           % total display duration (s)
fc = 500;            % carrier / modulation frequency (Hz)
fv = 8;              % low-frequency envelope frequency (Hz)

t = (0:1/fs:T).';

% low-frequency envelope (conceptual throat vibration / motion)
env = 0.55 + 0.25*sin(2*pi*fv*t) + 0.05*sin(2*pi*2*fv*t);

% reference square wave: LED ON/OFF
ref = square(2*pi*fc*t);
ref_bin = ref > 0;

% raw sensor signal:
% baseline + modulated optical response + small ripple/noise-like term
raw = 1.0 + env.* (0.45*(ref_bin))...
    + 0.015*sin(2*pi*8*fc*t)...
    + 0.008*sin(2*pi*13*fc*t);

% cycle-wise demodulation
rise_idx = find(diff(ref_bin)==1) + 1;

demod_t = [];
demod_y = [];
on_means = nan(size(raw));
off_means = nan(size(raw));

for k = 1:numel(rise_idx)-1
    i1 = rise_idx(k);
    i2 = rise_idx(k+1)-1;

    idx = i1:i2;
    rb  = ref_bin(idx);
    x   = raw(idx);

    on_idx  = idx(rb==1);
    off_idx = idx(rb==0);

    if isempty(on_idx) || isempty(off_idx)
        continue;
    end

    m_on  = mean(raw(on_idx));
    m_off = mean(raw(off_idx));
    dval  = m_on - m_off;

    on_means(on_idx)   = m_on;
    off_means(off_idx) = m_off;

    demod_t(end+1,1) = mean(t(idx)); %#ok<SAGROW>
    demod_y(end+1,1) = dval;         %#ok<SAGROW>
end

% smooth conceptual demod envelope for display only
demod_y_s = movmean(demod_y, 3);

%% ---------------- Figure ----------------
fig = figure('Color','w','Position',[100 100 1000 700]);

% ---------- 1. Reference ----------
ax1 = subplot(4,1,1);
hold on;

% shade LED ON phases
shade_phases(t, ref_bin, [0.92 0.92 0.92]);

plot(t, double(ref_bin), 'k', 'LineWidth', 1.5);
ylim([-0.2 1.2]);
yticks([0 1]);
yticklabels({'OFF','ON'});
xlim([t(1) t(end)]);
ylabel('Reference');
title('Demodulation illustration');
box off; grid on;

text(t(round(0.03*numel(t))), 1.05, 'LED ON', 'FontSize', 10, 'Color', [0.25 0.25 0.25]);
text(t(round(0.11*numel(t))), 0.10, 'LED OFF', 'FontSize', 10, 'Color', [0.25 0.25 0.25]);

% ---------- 2. Raw modulated signal ----------
ax2 = subplot(4,1,2);
hold on;

shade_phases(t, ref_bin, [0.95 0.95 0.95]);
plot(t, raw, 'Color', [0 0.4470 0.7410], 'LineWidth', 1.2);

xlim([t(1) t(end)]);
ylabel('Raw signal');
box off; grid on;

% ---------- 3. ON/OFF phase averaging ----------
ax3 = subplot(4,1,3);
hold on;

shade_phases(t, ref_bin, [0.95 0.95 0.95]);
plot(t, raw, 'Color', [0.75 0.85 0.95], 'LineWidth', 0.8);

% show phase means
plot(t, on_means,  '.', 'Color', [0.8500 0.3250 0.0980], 'MarkerSize', 8);
plot(t, off_means, '.', 'Color', [0.2 0.2 0.2], 'MarkerSize', 8);

xlim([t(1) t(end)]);
ylabel('Phase means');
legend({'raw','LED ON mean','LED OFF mean'}, 'Location','northeast');
box off; grid on;

% ---------- 4. Demodulated envelope ----------
ax4 = subplot(4,1,4);
hold on;

plot(demod_t, demod_y,   'o-', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8, 'MarkerSize', 3);
plot(demod_t, demod_y_s, '-',  'Color', [0.8500 0.3250 0.0980], 'LineWidth', 2.0);

xlabel('Time (s)');
ylabel('Envelope');
xlim([t(1) t(end)]);
legend({'cycle-wise demod','demodulated envelope'}, 'Location','northeast');
box off; grid on;

linkaxes([ax1 ax2 ax3 ax4], 'x');

%% ---------------- Optional save ----------------
exportgraphics(fig, 'Figure_demodulation_illustration.png', 'Resolution', 300);

%% ============================================================
%% Local function
%% ============================================================
function shade_phases(t, ref_bin, facecolor)
    ref_bin = logical(ref_bin(:));
    t = t(:);

    d = diff([0; ref_bin; 0]);
    starts = find(d==1);
    stops  = find(d==-1)-1;

    yl = ylim;
    for i = 1:numel(starts)
        xs = [t(starts(i)) t(stops(i)) t(stops(i)) t(starts(i))];
        ys = [yl(1) yl(1) yl(2) yl(2)];
        patch(xs, ys, facecolor,...
            'EdgeColor','none', 'FaceAlpha',0.4);
    end
    uistack(findobj(gca,'Type','patch'),'bottom');
end