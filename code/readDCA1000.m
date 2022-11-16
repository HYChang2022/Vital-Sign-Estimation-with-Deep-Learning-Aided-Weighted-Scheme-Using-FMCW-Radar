%%% This script is used to read the binary file produced by the DCA1000
%%% and Mmwave Studio
%%% Command to run in Matlab GUI -
% The copyright of the script belongs to Texas Instrument (TI)
% Some parts are rewritten by Po-Yen Lin, poyen.lin@outlook.com
function [retVal,numChirps] = readDCA1000(fileName,N,ADCS)
    %% global variables
    % change based on sensor config
    numADCSamples = ADCS; % number of ADC samples per chirp %default256
    numADCBits = 16; % number of ADC bits per sample
    numRX = N; % number of receivers %%default4
%     numLanes = 2; % do not change. number of lanes is always 2
    isReal = 0; % set to 1 if real only data, 0 if complex data
    %% read file
    % read .bin file
    fid = fopen(fileName,'r');
    adcData = fread(fid, 'int16');
    % if 12 or 14 bits ADC per sample compensate for sign extension
    if numADCBits ~= 16
        l_max = 2^(numADCBits-1)-1;
        adcData(adcData > l_max) = adcData(adcData > l_max) - 2^numADCBits;
    end
    fclose(fid);
    fileSize = size(adcData, 1);
    % real data reshape, filesize = numADCSamples*numChirps
    if isReal
        numChirps = fileSize/numADCSamples/numRX;
        LVDS = zeros(1, fileSize);
        %create column for each chirp
        LVDS = reshape(adcData, numADCSamples*numRX, numChirps);
        %each row is data from one chirp
        LVDS = LVDS.';
    else
        % for complex data
        % filesize = 2 * numADCSamples*numChirps
        adcDatao=adcData;
        adcData=adcDatao(1:end-mod(length(adcDatao),numADCSamples*2));
        fileSize = size(adcData, 1);
        numChirps = fileSize/2/numADCSamples/numRX;
        LVDS = zeros(1, fileSize/2);
        %combine real and imaginary part into complex data
        %read in file: 2I is followed by 2Q
        counter = 1;
        for i=1:4:fileSize-1
            LVDS(1,counter) = adcData(i) + sqrt(-1)*adcData(i+2); 
            LVDS(1,counter+1) = adcData(i+1)+sqrt(-1)*adcData(i+3);
            counter = counter + 2;
        end
        zzz=LVDS;
        %create column for each chirp
        LVDS = reshape(zzz, numADCSamples*numRX, numChirps);
        %each row is data from one chirp
        LVDS = LVDS.';
    end
    %organize data per RX
    adcData = zeros(numRX,numChirps*numADCSamples);
    for row = 1:numRX
        for i = 1: numChirps
            adcData(row, (i-1)*numADCSamples+1:i*numADCSamples) = LVDS(i, (row-1)*numADCSamples+1:row*numADCSamples);
        end
    end
    % return receiver data
    retVal = adcData;