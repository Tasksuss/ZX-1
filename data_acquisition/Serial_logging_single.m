%% ZX-1 DC acquisition (single channel, max throughput)
clear; clc;

%% ==== SETTINGS ====
port = "COM3";      % 改成你的端口
baud = 460800;
duration_s = 10;

filename = "DC_sentence_eight.mat";   % 自己改名字

%% ==== SERIAL SETUP ====
s = serialport(port, baud, "Timeout", 1);
flush(s);
pause(2);   % 等 Arduino reset

%% ==== BUFFER ====
% 单通道：2 bytes/sample
% 预估：~8000 samples/s → 16000 bytes/s → 10s ≈ 160kB
% 留余量
max_bytes = 20000 * duration_s;
buf = zeros(max_bytes, 1, 'uint8');
ptr = 1;

%% ==== ACQUISITION LOOP ====
t0 = tic;
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
clear s

%% ==== UNPACK ====
% 每 2 bytes = 1 sample
n_complete = floor(numel(raw)/2);
raw = raw(1:2*n_complete);

b = reshape(raw, 2, []);
sig = uint16(b(1,:)) + bitshift(uint16(b(2,:)), 8);

idx = 0:(numel(sig)-1);

%% ==== META ====
fs_est = numel(sig) / duration_s;

%% ==== SAVE ====
save(filename, "idx", "sig", "raw", "baud", "duration_s", "fs_est");

fprintf("Saved %d samples\n", numel(sig));
fprintf("Sample rate: %.2f Hz\n", fs_est);