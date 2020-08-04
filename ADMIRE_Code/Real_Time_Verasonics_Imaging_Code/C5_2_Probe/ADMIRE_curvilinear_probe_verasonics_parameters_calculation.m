% Copyright 2020 Christopher Khan

% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the license at

%     http://www.apache.org/licenses/LICENSE-2.0

% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and 
% limitations under the License.


% Description of
% ADMIRE_curvilinear_probe_verasonics_parameters_calculation.m:
% This script calculates some of the ADMIRE parameters for a Verasonics dataset
% that is acquired with a curvilinear probe


% Clear the workspace and close all figures
clear all; close all;

% Specify the path to the dataset and the filename
data_path = 'enter path here';
filename = 'Dataset_Curvilinear_Probe.mat';

% Load the data
load(fullfile(data_path, filename));

% Calculate some of the parameters that are required for generating the
% ADMIRE models
num_buffer_rows = size(RcvData, 1);
num_elements = P.numTx;
total_elements_on_probe = Trans.numelements;
num_beams = P.num_beams;
numRcvSamples = floor((Receive(1).endDepth - Receive(1).startDepth) .* 2 ...
    .* Receive(1).samplesPerWave);
t0 = round(((Receive(1).startDepth .* 2) + (Trans.lensCorrection .* 2) + TW(1).peak + ...
    max(TX(1).Delay)) .* (Receive(1).samplesPerWave ./ 2));
num_depths = length([t0:numRcvSamples]);
c = Resource.Parameters.speedOfSound;
f0 = Trans.frequency .* (1E6);
fs = Receive(1).decimSampleRate .* (1E6);
BW = 0.6; % This assumes the default bandwidth of 60% for the Verasonics system
probe_pitch = Trans.spacingMm ./ 1000;
probe_radius = Trans.radiusMm ./ 1000;