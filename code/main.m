% The author of the script is Po-Yen Lin, poyen.lin@outlook.com
clc;clear;close all;

%% This part is emp[oyed to extract ADC signal from radar raw data and acquire vital sign signal
ADCS = 80;antennaNumber=1;
[OriginalData,~]=readDCA1000('..\data\breathData\measure\train\rawData_22.bin',antennaNumber,ADCS);
rawData=OriginalData;
[w,RPmap,phaseSignal,heartSignal,breathSignal]=dataAcquisition(rawData,ADCS);
% w: FFT frequencies
% RPmap: range-time signal after 1-D FFT 
% phaseSignal: signal contains vital sign
% heartSignal: filtered phaseSignal in [0.8 2.0] Hz
% breathSignal: filtered phaseSignal in [0.1 0.5] Hz
%% Save or load
% save('C:\Users\123\Desktop\program\data\breathData\measure\train\rawData_22.mat','w','RPmap','phaseSignal','heartSignal','breathSignal');
%% Demonstration
load('..\data\heartData\measure\train\rawData_2.mat');
%[w,RPmap,phaseSignal,heartSignal,breathSignal]

%% Radar parameters
fsFast=2*10^6;
s=(3*10^8*5*10^(-5))/(2*4*10^9);
fsSlow=100;
%% The first 14 seconds is not used
p=phaseSignal(14*fsSlow+1:end);
h=heartSignal(14*fsSlow+1:end);
b=breathSignal(14*fsSlow+1:end);
T=(14*fsSlow:length(phaseSignal)-1)/fsSlow;
%% Plot range profile
figure,
[q,r]=meshgrid((14*fsSlow+1:length(RPmap))/fsSlow,((w/(2*pi)*fsFast*s)));
mesh(q,r,RPmap(:,14*fsSlow+1:end),'Facecolor','interp');
colorbar;colormap('jet');
xlabel('Time(Sec.)','FontName','consolas');ylabel('Distance(m)','FontName','consolas');zlabel('Strength','FontName','consolas');
view(0,90);
axis([14 194 0 3.721 -inf inf]);
title('Range Profile');
%% Phase signal
figure,
plot(T,p,'LineWidth',1.5);grid;
xlabel('Time(Sec.)','FontName','consolas'),ylabel('Amp.','FontName','consolas');
title('Phase Response');
axis([14 194 -inf inf]);
%% Heart signal and breath signal
figure,
subplot(211),
plot(T,h,'LineWidth',1.5);grid;
xlabel('Time(Sec.)','FontName','consolas'),ylabel('Amp.','FontName','consolas');
title('Heart Signal');
axis([14 194 -inf inf]);
subplot(212),
plot(T,b,'LineWidth',1.5);grid;
xlabel('Time(Sec.)','FontName','consolas'),ylabel('Amp.','FontName','consolas');
title('Breath Signal');
axis([14 194 -inf inf]);
