# Vital-Sign-Estimation-with-Deep-Learning-Aided-Weighted-Scheme-Using-FMCW-Radar
- Official repository of [Fast Acquisition and Accurate Vital Sign Estimation with Deep Learning-Aided Weighted Scheme Using FMCW Radar](https://ieeexplore.ieee.org/document/9860799)
- This repository contains the code and the dataset
## About
- The code directory contains MATLAB scripts for data preprocessing
- The data directory contains the experiment results, which can be further divided into breathData and heartData. 
   - ground_truth directory contains the reference label for the respiration rate and heart rate 
   - measure directory contains the received radar signal
   - The duration of radar signal is of 180 seconds, the reference label is recorded with a sliding window size of six seconds and an overlap of five seconds
- The DAWS_algorithm_code directory contains the implemented code of our work as well as the preprocessing for dataset.
   - First copy "rawData_* .bin" files  from data directories and run the "DAWS_algorithm_code\data preprocessing\bin to mat\bin_to_mat.m" code to acquire "radarSignal_* .mat" files. Note that the input and output settings of files in "bin_to_mat.m" might require manual set. 
   - Unzip "DAWS_algorithm_code\data\data_beat_test\breathing\ground_truth.zip". Note that there are four zipped files in total.
   - Run "DAWS_algorithm_code\RPM_generator.m"
   - Now you are ready to train DAWS with "DAWS_algorithm_code\CNN_breath.py"
## Citation
```
@INPROCEEDINGS{2022_Chang,
  author={Chang, Hsin-Yuan and Hsu, Chih-Hsuan and Chung, Wei-Ho},
  booktitle={2022 IEEE 95th Vehicular Technology Conference: (VTC2022-Spring)}, 
  title={Fast Acquisition and Accurate Vital Sign Estimation with Deep Learning-Aided Weighted Scheme Using FMCW Radar}, 
  year={2022},
  pages={1-6}}
```
