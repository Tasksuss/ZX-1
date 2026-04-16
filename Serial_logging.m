clear; clc;

port = "COM3";        % 改成你的端口
baud = 460800;
duration_s = 30;

s = serialport(port, baud, "Timeout", 1);
flush(s);
pause(2);   % 给 Uno 重置一点时间

% 预估容量：假设先按 4000 frames/s * 4 bytes/frame * 10 s，再留余量
max_bytes = 4500 * 4 * duration_s;
buf = zeros(max_bytes, 1, 'uint8');
ptr = 1;

t0 = tic;
while toc(t0) < duration_s
    n = s.NumBytesAvailable;
    if n > 0
        data = read(s, n, "uint8");   % 直接按字节抓
        k = numel(data);

        if ptr + k - 1 > numel(buf)
            % 扩容
            buf = [buf; zeros(max_bytes,1,'uint8')];
        end

        buf(ptr:ptr+k-1) = data;
        ptr = ptr + k;
    end
end

raw = buf(1:ptr-1);
clear s

% 丢掉末尾不完整帧
n_complete = floor(numel(raw)/4);
raw = raw(1:4*n_complete);

%% 解包：little-endian
b = reshape(raw, 4, []);
ref = uint16(b(1,:)) + bitshift(uint16(b(2,:)), 8);
sig = uint16(b(3,:)) + bitshift(uint16(b(4,:)), 8);
fs = numel(ref)/duration_s;
idx = 0:(numel(ref)-1);
%%
save("map_fifth_speech_rep4.mat", "idx", "ref", "sig", "raw", "baud", "duration_s", "fs");

fprintf("Saved %d complete frames to dual_channel_log.mat\n", numel(ref));
fprintf("Frame rate: %.2f frames/s\n", numel(ref)/duration_s);