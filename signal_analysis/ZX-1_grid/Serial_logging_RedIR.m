%% ZX-1 alternating RED/IR PPG acquisition and demux
clear; clc; close all;

%% ==== SETTINGS ====
port = "COM3";          % change this
baud = 460800;
duration_s = 30;        % e.g. 20 seconds
filename = "ppg_alt_red_ir5.mat";

led_rate_hz = 350;      % must match Arduino
slot_rate_hz = 4 * led_rate_hz;

%% ==== SERIAL SETUP ====
s = serialport(port, baud, "Timeout", 1);
flush(s);
pause(2);   % allow Arduino reset

%% ==== BUFFER ====
% Packet = 5 bytes
% At 400 packets/s and 20 s => 8000 packets => 40000 bytes
% leave margin
max_bytes = 100000;
buf = zeros(max_bytes, 1, 'uint8');
ptr = 1;

%% ==== ACQUIRE ====
t0 = tic;
fprintf('Start Signal Acquisition')
while toc(t0) < duration_s
    n = s.NumBytesAvailable;
    if n > 0
        data = read(s, n, "uint8");
        k = numel(data);

        if ptr + k - 1 > numel(buf)
            buf = [buf; zeros(max_bytes,1,'uint8')];
        end

        buf(ptr:ptr+k-1) = data;
        ptr = ptr + k;
    end
end

raw = buf(1:ptr-1);
fprintf('End Signal Acquisition')

clear s

%% ==== PARSE PACKETS ====
% Packet format:
% [0xAA][0x55][state][sig_low][sig_high]

hdr1 = uint8(170);  % 0xAA
hdr2 = uint8(85);   % 0x55

states = [];
sig = [];

i = 1;
Nraw = numel(raw);

while i <= Nraw - 4
    if raw(i) == hdr1 && raw(i+1) == hdr2
        st = raw(i+2);
        val = uint16(raw(i+3)) + bitshift(uint16(raw(i+4)), 8);

        states(end+1,1) = st; %#ok<SAGROW>
        sig(end+1,1) = val;   %#ok<SAGROW>

        i = i + 5;
    else
        i = i + 1;
    end
end

n_packets = numel(sig);
fprintf('Parsed %d packets\n', n_packets);
fprintf('Packet rate: %.2f Hz\n', n_packets / duration_s);

t_slot = (0:n_packets-1)' / slot_rate_hz;

%% ==== CHECK STATE DISTRIBUTION ====
fprintf('State counts:\n');
for st = 0:3
    fprintf('  state %d: %d\n', st, sum(states == st));
end

%% ==== DEMUX STATES ====
% 0 = RED ON
% 1 = OFF after RED
% 2 = IR ON
% 3 = OFF after IR

red_all = double(sig(states == 0));
offR_all = double(sig(states == 1));
ir_all = double(sig(states == 2));
offIR_all = double(sig(states == 3));

N = min([numel(red_all), numel(offR_all), numel(ir_all), numel(offIR_all)]);

red = red_all(1:N);
offR = offR_all(1:N);
ir = ir_all(1:N);
offIR = offIR_all(1:N);

% corrected channels
red_corr = red - offR;
ir_corr  = ir  - offIR;

t_led = (0:N-1)' / led_rate_hz;

%% ==== OPTIONAL LIGHT SMOOTHING ====
% just for viewing, not mandatory
red_s = movmean(red_corr, 100);
ir_s  = movmean(ir_corr, 100);

%% ==== SAVE ====
meta.port = port;
meta.baud = baud;
meta.duration_s = duration_s;
meta.led_rate_hz = led_rate_hz;
meta.slot_rate_hz = slot_rate_hz;
meta.packet_format = '[0xAA][0x55][state][sig_low][sig_high]';
meta.state_map = '0=RED,1=OFF_after_RED,2=IR,3=OFF_after_IR';

save(filename, ...
    'raw', 'states', 'sig', 't_slot', ...
    'red', 'offR', 'ir', 'offIR', 'red_corr', 'ir_corr', 't_led', 'meta');

fprintf('Saved %s\n', filename);

%% ==== PLOTS ====

% raw slot-level signal
figure;
plot(t_slot, double(sig), 'LineWidth', 1);
xlabel('Time (s)');
ylabel('ADC count');
title('Raw slot-level signal');

% corrected channels
figure;
plot(t_led, red_corr, '.-'); hold on;
plot(t_led, ir_corr, '.-');
xlabel('Time (s)');
ylabel('Corrected signal (ADC count)');
title('Corrected RED and IR channels');
legend('red\_corr', 'ir\_corr');

% smoothed corrected channels
figure;
plot(t_led, red_s, 'LineWidth', 1.2); hold on;
plot(t_led, ir_s, 'LineWidth', 1.2);
xlabel('Time (s)');
ylabel('Corrected signal (smoothed)');
title('Smoothed corrected RED and IR channels');
legend('red\_corr smoothed', 'ir\_corr smoothed');

% Quick PPG bandpass: 0.5 - 8 Hz
fs = led_rate_hz;

[b,a] = butter(3, [0.5 8]/(fs/2), 'bandpass');

red_ppg = filtfilt(b, a, red_s);
ir_ppg  = filtfilt(b, a, ir_s);

figure;
plot(t_led, red_ppg, 'LineWidth', 1.2); hold on;
plot(t_led, ir_ppg, 'LineWidth', 1.2);
xlabel('Time (s)');
ylabel('Bandpassed signal');
title('Bandpassed RED and IR PPG-like signals');
legend('red\_ppg', 'ir\_ppg');