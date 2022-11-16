% The author of the script is Po-Yen Lin, poyen.lin@outlook.com

function [w,RPmap,phaseSignal,heartSignal,breathSignal]=dataAcquisition(rawData,ADCS)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
Data=rawData;
fsFast=2*10^6;
FP=128;
X=zeros(FP,length(Data)/ADCS);

for ii=1:length(Data)/ADCS
    [X(:,ii),w]=FFTNonshift(window(@nuttallwin,ADCS).*Data(1,1+(ii-1)*ADCS:ADCS+(ii-1)*ADCS).',FP);
end

RPmap=abs(X)/(2*pi);

[~,Posi]=max(abs(X(1:end,:)));
pos=mode(Posi');

fsSlow=100;
MaxRow=X(pos,:);
fc = 77e9;
c = physconst('LightSpeed');
lambda = c/fc;

phaseSignal=unwrap(angle(MaxRow))*lambda/(4*pi);

%% heart signal 
Wn =[0.8 2.0]/(fsSlow/2);
[a,b] = butter(3, Wn);
filterSignal_heart = filter(a,b,phaseSignal);
heartSignal = filterSignal_heart;
%% breath signal

Wn =[0.1 0.5]/(fsSlow/2);
[a,b] = butter(3, Wn);
filterSignal_breath = filter(a,b,phaseSignal);
breathSignal = filterSignal_breath;
end
