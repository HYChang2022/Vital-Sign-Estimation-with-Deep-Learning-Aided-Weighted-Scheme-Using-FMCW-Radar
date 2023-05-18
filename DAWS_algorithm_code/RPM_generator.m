%% test
% clc;
clear;
close all;

%% Load data
load_beat_signal = 1;
training_or_testing = 0;     % 0: training data,  1: testing data
breathing_or_heartbeat = 0;  % 0: breathing data, 1: heartbeat data

%% Set up
selected_bin_num = 5;  % select 5 range bins
fft_num = 80;          % according to the number of sampling points of a chirp of radar DCA1000
print_period = 50;

if training_or_testing == 0 && breathing_or_heartbeat == 0
    data_num = 22;
elseif training_or_testing == 0 && breathing_or_heartbeat == 1
    data_num = 16;
elseif training_or_testing == 1
    data_num = 10;
end
window_num = 175;
% window_num is equal to 175 = 180 - 6 + 1 (180 sec signal sliding with 6 sec window, each sliding step is 1 sec)

plot_EWS_CDF = 1;
plot_check_idx = [0];
plot_after_bandpass_filter_idx = [0];
plot_M1_idx = [0];
plot_EWS_idx = [0];
plot_range_map_idx = [0];
%% Parameters
frequency_carrier = 77 * 10^9;
BW = 4*10^9;                      % bandwidth of FMCW radar
speed_of_light = physconst('LightSpeed');
delta_R = speed_of_light/(2*BW);   % range resolution
ObsTime = 6;                       % unit: second
FramePeriod = 10*10^(-3);          % chirp period = 10 ms
slow_time_fre = 1/FramePeriod;
FrameNum = ObsTime/FramePeriod;

chirp_duration = 50*10^(-6) * 0.8; % chirp duration = 50 us; because radar_ADC_samples = 80 => 50*0.8;
fs = 2*10^6;                       % fast time sampling frequency -> sampling period = 0.5 us
chirp_samples_num = chirp_duration/(1/fs);

freq_breathing_lower_bound = 0.1;
freq_breathing_upper_bound = 0.5;
freq_heartbeat_lower_bound = 0.8;
freq_heartbeat_upper_bound = 2.0;
filter_transition_rate = 0.1;
%% Initialization
selected_range_bin_cell = zeros(selected_bin_num,FrameNum);

data_bin = zeros(window_num*data_num, slow_time_fre*ObsTime, selected_bin_num, 3);
data_est = zeros(window_num*data_num, 2, 5);
data_ref = zeros(window_num*data_num, 2);

t_start = tic;
%%
for data_idx = 1:data_num
    if training_or_testing == 0 && breathing_or_heartbeat == 1 && data_idx == 11 % heartbeat的train的第11個data檔有問題(中間有一段都是0)
        continue
    end
        
    if load_beat_signal == 1 && training_or_testing == 0 && breathing_or_heartbeat == 0
        load(sprintf('./data/data_beat_train/breathing/radarSignal_%d.mat', data_idx)); % beat signal from radar
        load(sprintf('./data/data_beat_train/breathing/ground truth/rawData_%d.mat', data_idx)); % ground truth frequency from contact devices
    elseif load_beat_signal == 1 && training_or_testing == 0 && breathing_or_heartbeat == 1
        load(sprintf('./data/data_beat_train/heartbeat/radarSignal_%d.mat', data_idx));
        load(sprintf('./data/data_beat_train/heartbeat/ground truth/rawData_%d.mat', data_idx));
    elseif load_beat_signal == 1 && training_or_testing == 1 && breathing_or_heartbeat == 0
        load(sprintf('./data/data_beat_test/breathing/radarSignal_%d.mat', data_idx));
        load(sprintf('./data/data_beat_test/breathing/ground truth/rawData_%d.mat', data_idx));
    elseif load_beat_signal == 1 && training_or_testing == 1 && breathing_or_heartbeat == 1
        load(sprintf('./data/data_beat_test/heartbeat/radarSignal_%d.mat', data_idx));
        load(sprintf('./data/data_beat_test/heartbeat/ground truth/rawData_%d.mat', data_idx));
    end
    %% get range profile matrix from the data
    beat_signal_vector = rawData;
    radar_ADC_samples = 80;
    reservation_time = 180;  % The data at the beginning time is unstable, so we only keep the latter part (180 sec).
    start_time = (length(beat_signal_vector)/radar_ADC_samples - reservation_time*slow_time_fre) / slow_time_fre;

    chirp_total_num = length(beat_signal_vector)/radar_ADC_samples;
    beat_signal_matrix_total = zeros(radar_ADC_samples,chirp_total_num);
    range_profile_matrix_total = zeros(fft_num,chirp_total_num);

    for chirp_idx = 1:chirp_total_num
        beat_signal_matrix_total(:,chirp_idx) = beat_signal_vector(1,1+(chirp_idx-1)*radar_ADC_samples:chirp_idx*radar_ADC_samples);
        range_profile_matrix_total(:,chirp_idx) = fft(window(@nuttallwin,radar_ADC_samples).*beat_signal_matrix_total(:,chirp_idx),fft_num);
        % range_profile_matrix_total(:,chirp_idx) = fft(beat_signal_matrix_total(:,chirp_idx),fft_num);
    end
    beat_signal_matrix = beat_signal_matrix_total(:,start_time/FramePeriod+1:end); % begin from start_time
    range_profile_matrix = range_profile_matrix_total(:,start_time/FramePeriod+1:end);

    for map_idx = 1:((chirp_total_num-start_time/FramePeriod) - FrameNum) / slow_time_fre + 1 % Divided into each map (which contains ObsTime sec)
        beat_signal_matrix_seg(map_idx,:,:) = beat_signal_matrix(:,1+(map_idx-1)*slow_time_fre:FrameNum+(map_idx-1)*slow_time_fre);
        range_profile_matrix_seg(map_idx,:,:) = range_profile_matrix(:,1+(map_idx-1)*slow_time_fre:FrameNum+(map_idx-1)*slow_time_fre);
    end

    %% load reference frequency (regarded as ground truth) & calculate the mean value of the window (a window corresponds to a range map)
    window_num = ((chirp_total_num-start_time/FramePeriod) - FrameNum) / slow_time_fre + 1;
    gt_mean_value = zeros(1,window_num);
    for window_idx = 1:window_num
        if breathing_or_heartbeat == 0
            gt_mean_value = res;                      % mean value of breathing data has been calculated by ref_res.m
            frequency_breathing = gt_mean_value / 60; % BPM to Hz
            frequency_heartbeat = zeros(size(gt_mean_value));
        elseif breathing_or_heartbeat == 1
            gt_mean_value(1,window_idx) = mean(grd_truth(window_idx:window_idx+ObsTime-1));
            frequency_breathing = zeros(size(gt_mean_value));
            frequency_heartbeat = gt_mean_value / 60;
        end
    end
    
    % reference frequency (for model input)
    if breathing_or_heartbeat == 0
        data_ref(1+window_num*(data_idx-1):175+window_num*(data_idx-1),1) = frequency_breathing * 60;
    elseif breathing_or_heartbeat == 1
        data_ref(1+window_num*(data_idx-1):175+window_num*(data_idx-1),2) = frequency_heartbeat * 60;
    end

    %% RangeMaps
    for map_idx = 1:window_num
        if rem(map_idx,print_period) == 0 % && (sum(map_idx == plot_spectrum_idx) ~= 1)
            fprintf('---------------\n data index = %d/%d; map index = %d \n',data_idx,data_num,map_idx);
        end

        %% range profile matrix & phase extraction
        % range-profile matrix
        RangeMap = squeeze(range_profile_matrix_seg(map_idx,:,:));

        % phase extraction (estimated vital signs signal)
        [~, posi] = max(abs(RangeMap));
        table_posi = tabulate(posi);
        [num,posi_value] = sort(table_posi(:,2),'descend');
        condition_1 = abs(posi_value(2)-posi_value(3)) < 3 && (num(2)+num(3)) > num(1);  % they are neighbors & the sum is greater
        condition_2 = abs(posi_value(2)-posi_value(3)) < 3 && abs(posi_value(3)-posi_value(4)) < 3 && (num(2)+num(3)+num(4)) > num(1);
        if condition_1 || condition_2
            posi_RangeMap = posi_value(2);
        elseif posi_value(1) == 2
            posi_RangeMap = posi_value(2);
        else
            posi_RangeMap = posi_value(1);
        end
        phase_beat_signal(1,:) = unwrap(angle(RangeMap(posi_RangeMap,:)));
        esti_range_variation(1,:) = phase_beat_signal(1,:) * speed_of_light/(4*pi*frequency_carrier);

        if sum(map_idx == plot_check_idx) == 1
            figure_yrange = fft_num;
            figure()
            imagesc(0:ObsTime, (0:figure_yrange-1)*delta_R, abs(RangeMap));
            title('Range Profile Matrix'); ylabel('Range (m)'); xlabel('Time (sec)');
        end

        %% remove high frequency noise by 2Hz low-pass filter (priori knowledge: the highest frequency of vital signs is 2Hz)
        Wn = 2 * (1+filter_transition_rate) / (slow_time_fre/2);
        [filter_LP_b,filter_LP_a] = butter(5,Wn,'low');
        esti_range_variation_LPF = filter(filter_LP_b,filter_LP_a,esti_range_variation);
        %% remove DC by high-pass filter
        Wn = 0.05 / (slow_time_fre/2);  % filter out the frequencies below 0.05 Hz
        [filter_HP_b,filter_HP_a] = butter(3,Wn,'high');
        esti_range_variation_HPF = filter(filter_HP_b,filter_HP_a,esti_range_variation_LPF);

        if sum(map_idx == plot_check_idx) == 1
            figure()
            subplot(311)
            plot(esti_range_variation); hold on;
            title('original'); xlabel('Chirp'); ylabel('Range (m)');
            legend('esti range variation');
            subplot(312)
            plot(esti_range_variation_LPF); hold on;
            title('after low-pass filter (2 Hz)'); xlabel('Chirp'); ylabel('Range (m)');
            legend('esti range variation');
            subplot(313)
            plot(esti_range_variation_HPF); hold on;
            title('after high-pass filter (remove DC)'); xlabel('Chirp'); ylabel('Range (m)');
            legend('esti range variation');
            sgtitle('in order to check')
        end

        %% estimation of the breathing/heartbeat frequency (by NOMP)
        esti_vital_sign_signal = esti_range_variation_HPF;

        % pass through separate band-pass filters for breathing & heartbeat
        Wn_b = [freq_breathing_lower_bound*(1-filter_transition_rate) freq_breathing_upper_bound] / (slow_time_fre/2);  % Breathing rate: 0.1 ~ 0.5 Hz
        [filter_b_b,filter_a_b] = butter(3,Wn_b,'bandpass');
        esti_vital_sign_signal_b = filter(filter_b_b,filter_a_b,esti_vital_sign_signal);

        Wn_h = [freq_heartbeat_lower_bound freq_heartbeat_upper_bound*(1+filter_transition_rate)] / (slow_time_fre/2);  % Heart rate: 0.8 ~ 2.0 Hz
        [filter_b_h,filter_a_h] = butter(3,Wn_h,'bandpass');
        esti_vital_sign_signal_h = filter(filter_b_h,filter_a_h,esti_vital_sign_signal);
        % the heartbeat needs to pass through an extra filter because the amplitude of breathing is too large
        esti_vital_sign_signal_h = filter(filter_b_h,filter_a_h,esti_vital_sign_signal_h);

        if sum(map_idx == plot_after_bandpass_filter_idx) == 1
            figure()
            subplot(211)
            plot(esti_vital_sign_signal_b); grid on; title('breathing');
            subplot(212)
            plot(esti_vital_sign_signal_h); grid on; title('heartbeat');
            sgtitle('esti signal after band-pass filter (time-domain)')
        end
        %% method 1: esti signal -> band-pass filter -> FFT peak (called M1)
        [~,posi] = max(abs(fft(esti_vital_sign_signal_b)));
        M1_esti_freq(1,1) = (posi-1)*slow_time_fre/length(esti_vital_sign_signal);  % est_b
        [~,posi] = max(abs(fft(esti_vital_sign_signal_h)));
        M1_esti_freq(1,2) = (posi-1)*slow_time_fre/length(esti_vital_sign_signal);  % est_h

        if sum(map_idx == plot_M1_idx) == 1
            fprintf('M4: \n ground truth: '); fprintf('%g  %g', frequency_breathing(map_idx), frequency_heartbeat(map_idx)); fprintf('\n');
            disp(M1_esti_freq);
            FFT_abs_FFT_h = abs(fft(esti_vital_sign_signal_h));
            fre_domain_FFT_h = (0:length(esti_vital_sign_signal_h)-1)*slow_time_fre/length(esti_vital_sign_signal_h);
            figure()
            plot(fre_domain_FFT_h,log10(FFT_abs_FFT_h/max(FFT_abs_FFT_h))); hold on;
            title('Spectrum of esti-vital sign signal 【after heartbeat bandpass filter】');
            legend('FFT'); grid on; xlim([0 2.2]);
        end
        %% method 2: esti signal -> NOMP (called M2)
        % parameter of NOMP
        S = eye(length(esti_vital_sign_signal));
        N_for_tau = 256;
        p_fa_for_tau = 1e-2;           % we choose tau to ensure that the false alarm rate is fixed
        sigma_for_tau = 1;
        tau = sigma_for_tau^2 * ( log(N_for_tau) - log( log(1/(1-p_fa_for_tau)) ) );
        % FFT case:
        y = esti_vital_sign_signal.' * 15000; % amplitude will affect the estimation of NOMP, so we amplify it
        [omega_est, ~, ~] = f_extractSpectrum(y, S, tau);
        freq_est = (omega_est/(2*pi) * slow_time_fre ).';
        
        find_breathing = find((freq_est >= freq_breathing_lower_bound).*(freq_est <= freq_breathing_upper_bound));
        if  isempty(find_breathing)
            M2_esti_freq(1,1) = M1_esti_freq(1,1);
        else
            M2_esti_freq(1,1) = sum(freq_est(find_breathing)) / length(find_breathing);
        end
        find_heartbeat = find((freq_est >= freq_heartbeat_lower_bound).*(freq_est <= freq_heartbeat_upper_bound));
        if  isempty(find_heartbeat)
            M2_esti_freq(1,2) = M1_esti_freq(1,2);
        else
            M2_esti_freq(1,2) = sum(freq_est(find_heartbeat)) / length(find_heartbeat);
        end
        
        %% Estimate the frequency of each bin (for EWS & input data of CNN model)
        % FFT case:
        for selected_idx = 1:1:selected_bin_num
            selected_range_bin_cell(selected_idx,:) = RangeMap(posi_RangeMap-(selected_bin_num+1)/2+selected_idx,:);
            selected_phase_beat_signal(selected_idx,:) = unwrap(angle(selected_range_bin_cell(selected_idx,:)));
            selected_esti_range_variation(selected_idx,:) = selected_phase_beat_signal(selected_idx,:) * speed_of_light/(4*pi*frequency_carrier);
            esti_range_variation_LPF = filter(filter_LP_b,filter_LP_a,selected_esti_range_variation(selected_idx,:));
            esti_range_variation_HPF(selected_idx,:) = filter(filter_HP_b,filter_HP_a,esti_range_variation_LPF);
            % M1: band-pass filter & fft peak
            esti_vital_sign_signal_b = filter(filter_b_b,filter_a_b,esti_range_variation_HPF(selected_idx,:));
            esti_vital_sign_signal_h = filter(filter_b_h,filter_a_h,esti_range_variation_HPF(selected_idx,:));
            esti_vital_sign_signal_h = filter(filter_b_h,filter_a_h,esti_vital_sign_signal_h);
            [~,posi_b] = max(abs(fft(esti_vital_sign_signal_b)));
            selected_M1_esti_freq(selected_idx,1) = (posi_b-1)*slow_time_fre/length(esti_vital_sign_signal_b);  % est_b
            [~,posi_h] = max(abs(fft(esti_vital_sign_signal_h)));
            selected_M1_esti_freq(selected_idx,2) = (posi_h-1)*slow_time_fre/length(esti_vital_sign_signal_h);  % est_h
            % M2: NOMP
            y = esti_range_variation_HPF(selected_idx,:).' * 15000;
            [omega_est, ~, ~] = f_extractSpectrum(y, S, tau);
            freq_est = (omega_est/(2*pi) * slow_time_fre ).';
            find_breathing = find((freq_est >= freq_breathing_lower_bound).*(freq_est <= freq_breathing_upper_bound));
            if  isempty(find_breathing)
                selected_M2_esti_freq(selected_idx,1) = selected_M1_esti_freq(selected_idx,1);
            else
                selected_M2_esti_freq(selected_idx,1) = sum(freq_est(find_breathing)) / length(find_breathing);
            end
            find_heartbeat = find((freq_est >= freq_heartbeat_lower_bound).*(freq_est <= freq_heartbeat_upper_bound));
            if  isempty(find_heartbeat)
                selected_M2_esti_freq(selected_idx,2) = selected_M1_esti_freq(selected_idx,2);
            else
                selected_M2_esti_freq(selected_idx,2) = sum(freq_est(find_heartbeat)) / length(find_heartbeat);
            end
            data_bin(map_idx + window_num*(data_idx-1),:,selected_idx,1) = abs(RangeMap(posi_RangeMap,:)).';
            data_bin(map_idx + window_num*(data_idx-1),:,selected_idx,2) = selected_phase_beat_signal(selected_idx,:).';
            data_bin(map_idx + window_num*(data_idx-1),:,selected_idx,3) = abs(fft(selected_phase_beat_signal(selected_idx,:))).';
            data_est(map_idx + window_num*(data_idx-1),1,selected_idx) = selected_M2_esti_freq(selected_idx,1) * 60;
            data_est(map_idx + window_num*(data_idx-1),2,selected_idx) = selected_M2_esti_freq(selected_idx,2) * 60;
        end
        EWS_M1_esti_freq = sum(selected_M1_esti_freq) / selected_bin_num;
        EWS_M2_esti_freq = sum(selected_M2_esti_freq) / selected_bin_num;
        
        % plot
        if sum(map_idx == plot_EWS_idx) == 1
            figure()
            subplot(211)
            plot(selected_esti_range_variation.'); grid on; title('esti range variations of selected bins');
            subplot(212)
            plot(esti_range_variation_HPF.'); grid on;
        end
        %% error of frequency estimation
        error_b(data_idx,1,map_idx) = abs(M2_esti_freq(1,1) - frequency_breathing(map_idx));
        error_h(data_idx,1,map_idx) = abs(M2_esti_freq(1,2) - frequency_heartbeat(map_idx));
        error_b_EWS(data_idx,1,map_idx) = abs(EWS_M2_esti_freq(1,1) - frequency_breathing(map_idx));
        error_h_EWS(data_idx,1,map_idx) = abs(EWS_M2_esti_freq(1,2) - frequency_heartbeat(map_idx));
        %% big error detection
        if sum(map_idx == plot_range_map_idx) == 1
            figure_yrange = fft_num;
            figure()
            imagesc(0:ObsTime, (0:figure_yrange-1)*delta_R, abs(RangeMap(1:figure_yrange,:)))
            title('Range Profile Matrix'); ylabel('Range (m)'); xlabel('Time (sec)');
        end
    end
end

%% error CDF
% flatten
error_b_vector(1,:) = reshape(error_b(:,1,:),1,[]);  
error_h_vector(1,:) = reshape(error_h(:,1,:),1,[]);
error_b_EWS_vector = reshape(error_b_EWS(:,1,:),1,[]);
error_h_EWS_vector = reshape(error_h_EWS(:,1,:),1,[]);

% remove the zeros because they were generated due to a problem with the 11th record of the heartbeat training data.
temp_1 = error_h_vector(1,:); temp_1(temp_1==0) = [];
clear error_h_vector; error_h_vector(1,:) = temp_1;
error_h_EWS_vector(error_h_EWS_vector==0) = [];

% CDF
[CDFvalue_err_breathing, err_for_CDF_breathing] = ecdf(error_b_vector(1,:) * 60); % fre -> BPM
[CDFvalue_err_heartbeat, err_for_CDF_heartbeat] = ecdf(error_h_vector(1,:) * 60);
[CDFvalue_err_breathing_EWS, err_for_CDF_breathing_EWS] = ecdf(error_b_EWS_vector * 60);
[CDFvalue_err_heartbeat_EWS, err_for_CDF_heartbeat_EWS] = ecdf(error_h_EWS_vector * 60);

if plot_EWS_CDF == 1 && breathing_or_heartbeat == 0
    figure()
    plot(err_for_CDF_breathing, CDFvalue_err_breathing, 'linewidth',1.5); hold on;
    plot(err_for_CDF_breathing_EWS, CDFvalue_err_breathing_EWS, 'linewidth',1.5);
    xlabel('error (BPM)'); ylabel('probability'); ylim([0 1.05]); grid on;
    title('error CDF (breathing)'); legend('original','EWS');
elseif plot_EWS_CDF == 1 && breathing_or_heartbeat == 1
    figure()
    plot(err_for_CDF_heartbeat, CDFvalue_err_heartbeat, 'linewidth',1.5); hold on;
    plot(err_for_CDF_heartbeat_EWS, CDFvalue_err_heartbeat_EWS, 'linewidth',1.5);
    xlabel('error (BPM)'); ylabel('probability'); ylim([0 1.05]); grid on;
    title('error CDF (heartbeat)'); legend('original','EWS');
end

%% Data used as input for the CNN model (which is written as python code)
data_ref_FFT = data_ref;
if training_or_testing == 0 && breathing_or_heartbeat == 0
    save('./data/data_for_model_input/breathing/data_feature_bin_train.mat','data_bin');
    save('./data/data_for_model_input/breathing/data_esti_freq_train.mat','data_est');
    save('./data/data_for_model_input/breathing/data_ref_train.mat','data_ref');
elseif training_or_testing == 0 && breathing_or_heartbeat == 1
    save('./data/data_for_model_input/heartbeat/data_feature_bin_train.mat','data_bin');
    save('./data/data_for_model_input/heartbeat/data_esti_freq_train.mat','data_est');
    save('./data/data_for_model_input/heartbeat/data_ref_train.mat','data_ref');
elseif training_or_testing == 1 && breathing_or_heartbeat == 0
    save('./data/data_for_model_input/breathing/data_feature_bin_test.mat','data_bin');
    save('./data/data_for_model_input/breathing/data_esti_freq_test.mat','data_est');
    save('./data/data_for_model_input/breathing/data_ref_test.mat','data_ref');
elseif training_or_testing == 1 && breathing_or_heartbeat == 1
    save('./data/data_for_model_input/heartbeat/data_feature_bin_test.mat','data_bin');
    save('./data/data_for_model_input/heartbeat/data_esti_freq_test.mat','data_est');
    save('./data/data_for_model_input/heartbeat/data_ref_test.mat','data_ref');
end

elapsed_time = toc(t_start);
fprintf('Elapsed time is %.2f seconds (%.2f minutes).\n', elapsed_time, elapsed_time/60);





