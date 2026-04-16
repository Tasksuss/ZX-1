%% ZX-1 data collection script: 10 s trials with beeps, save each trial as .mat
clear; clc;

%% ==== USER SETTINGS ====
port = "COM3";          % change this
baud = 460800;
base_folder = "ZX1_data";
session_name = input('Session name: ', 's');   % e.g. 2026-04-01_apollo_lateral500
duration_s = 10;
fs_nominal = 4415.9;    % optional note only
carrier_mode = "500Hz"; % or "DC"
sensor_id = "breadboard";
location = "lateral";

conditions = { ...
    'dark_led_off', ...
    'dark_led_on', ...
    'arm_static', ...
    'arm_press', ...
    'neck_static', ...
    'neck_jaw', ...
    'neck_a'};

n_repeats = 3;

%% ==== SETUP FOLDER ====
session_folder = fullfile(base_folder, session_name);
if ~exist(session_folder, 'dir')
    mkdir(session_folder);
end

%% ==== SERIAL ====
s = serialport(port, baud, "Timeout", 1);
flush(s);
pause(2);   % allow Uno reset

%% ==== BEEP SETTINGS ====
Fs_audio = 44100;
beep_dur = 0.12;
f_beep = 780;
tt = 0:1/Fs_audio:beep_dur;
beep_sig = 0.4 * sin(2*pi*f_beep*tt);

%% ==== RUN TRIALS ====
for c = 1:numel(conditions)
    cond = conditions{c};

    for r = 1:n_repeats
        fprintf('\n====================================\n');
        fprintf('Condition: %s | Repeat: %d/%d\n', cond, r, n_repeats);
        fprintf('0-2 s: rest | 2-8 s: task | 8-10 s: rest\n');
        fprintf('Beep at 0 s, 2 s, 8 s\n');
        input('Press ENTER when ready...','s');

        flush(s);

        % preallocate byte buffer
        max_bytes = 250000;   % expand if needed
        buf = zeros(max_bytes, 1, 'uint8');
        ptr = 1;

        t0 = tic;
        beep0_done = false;
        beep2_done = false;
        beep8_done = false;

        while toc(t0) < duration_s
            elapsed = toc(t0);

            % beeps
            if ~beep0_done && elapsed >= 0
                sound(beep_sig, Fs_audio);
                beep0_done = true;
            end
            if ~beep2_done && elapsed >= 2
                sound(beep_sig, Fs_audio);
                beep2_done = true;
            end
            if ~beep8_done && elapsed >= 8
                sound(beep_sig, Fs_audio);
                beep8_done = true;
            end

            % read available bytes
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

        % keep complete frames only: 4 bytes/frame = uint16 ref + uint16 sig
        n_complete = floor(numel(raw)/4);
        raw = raw(1:4*n_complete);

        b = reshape(raw, 4, []);
        ref = uint16(b(1,:)) + bitshift(uint16(b(2,:)), 8);
        sig = uint16(b(3,:)) + bitshift(uint16(b(4,:)), 8);
        idx = 0:(numel(ref)-1);

        meta.session_name = session_name;
        meta.condition = cond;
        meta.repeat = r;
        meta.duration_s = duration_s;
        meta.baud = baud;
        meta.carrier_mode = carrier_mode;
        meta.sensor_id = sensor_id;
        meta.location = location;
        meta.port = port;
        meta.nominal_frame_rate = numel(ref)/duration_s;
        meta.time_structure = '2-6-2';
        meta.beep_times_s = [0 2 8];

        fname = sprintf('%s_%s_rep%d.mat', session_name, cond, r);
        save(fullfile(session_folder, fname), 'idx', 'ref', 'sig', 'raw', 'meta');

        fprintf('Saved %s\n', fname);
        fprintf('Frames: %d | Frame rate: %.2f Hz\n', numel(ref), numel(ref)/duration_s);
    end
end

clear s
fprintf('\nAll trials complete.\n');