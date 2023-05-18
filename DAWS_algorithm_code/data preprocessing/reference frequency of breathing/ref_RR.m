clear all
close all
% breath=zeros(2,1);
ObsTime = 10; %unit: second
seg=10; %slow sampling rate

Nfft=ObsTime*seg;
time_scale=[0:1/seg:ObsTime-1/seg];
freq_scale=[0:1/Nfft*seg:seg-1/Nfft];

fileName = './breath03.mat';
load(fileName);
% figure()
% plot(breath);
jj=1;
for ii=1:seg:length(breath)-ObsTime*seg+1
    y=breath(ii:ii+ObsTime*seg-1,1);
    y=y-mean(breath(ii:ii+ObsTime*seg-1,1))*ones(ObsTime*seg,1);
%     figure
%     plot(time_scale,y)
%     figure
%     plot(freq_scale,abs(fft(y)))
%     pause(0.1)
    %% FFT
    fft_p = abs(fft(y));
    [pks,locs] = findpeaks(fft_p(1:length(fft_p)/2));
    f1=(locs-1)/Nfft*seg;
    index_res1=(f1<=0.5).*(f1>=0.1);
    index_heart1=(f1<=2).*(f1>=0.8);
    if sum(pks.*index_res1)==0
        est_res1(jj,1)=0;
    else
        [b1,res1] = max(pks.*index_res1);
        est_res1(jj,1)=(locs(res1)-1)/ObsTime;
    end

    %%  using NOMP algorithm to estimate freq.
    N=length(y);
    y=y/(mean(abs(y)));
    len=length(y); % number of measurements = N
    S = eye(len);
    p_fa = 1e-2;  % we choose tau to ensure that the false alarm rate is fixed:
    sigma=1;
    tau = sigma^2 * ( log(N) - log( log(1/(1-p_fa)) ) );
    [omega_est, gain_est, residue] = f_extractSpectrum(y, S, tau);
    omega_est=omega_est*seg;
    freq_est=(omega_est/(2*pi)).';
    find_r=find((freq_est<=0.5).*(freq_est>=0.1));

    if length(find_r)>1
        res(jj,1)=sum(freq_est(find_r))/length(find_r);
    elseif length(find_r)==0
        res(jj,1)=0;
    else
        res(jj,1)=freq_est(find_r);
    end
    jj=jj+1;
end
res=res*60;
save('./../../data/data_beat_train/breathing/rawData_1.mat','res');
clear res y
