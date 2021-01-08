% Copyright 2020 Christopher Khan, Kazuyuki Dei, Siegfried Schlunk, and
% Brett Byram

% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the license at

%     http://www.apache.org/licenses/LICENSE-2.0

% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and 
% limitations under the License.


% Description of ADMIRE_models_application_main.m:
% This script is the main program that applies ADMIRE to a set of channel
% data


% Clear the workspace and close all figures
clear all; close all;


%% User-Defined Parameters %%
% Define the input parameters (the parameters in the User-Defined Parameters section are the only parameters that need to be changed by the user)
apply_params.models_load_path = 'enter path here';  % Path to the directory from which the generated ADMIRE models are loaded
apply_params.data_load_path = 'enter path here';  % Path to the directory in which the channel data is located
apply_params.filename = 'channel_data_for_ADMIRE.mat';  % Name of the file that contains the channel data
apply_params.processed_data_save_path = 'enter path here'; % Path to the directory in which the data processed with ADMIRE is saved
apply_params.processed_data_filename = 'ADMIRE_processed_data.mat';  % Name of the file to which the data processed with ADMIRE is saved
apply_params.processor_type = 'GPU';  % Specify whether the CPU implementation of ADMIRE or the GPU implementation will be used: Specify either 'CPU' or 'GPU' 
apply_params.probe_type = 'Linear';  % Type of transducer array used: Specify either 'Linear' or 'Curvilinear'
apply_params.data_type = 'Reshaped';  % Specify the type of data that is being processed: Specify either 'Reshaped' or 'Verasonics RF Buffer' ('Reshaped' means that the data is in the form (Depths + t0 - 1) x Elements per Beam x Beams x Frames, and 'Verasonics RF Buffer' means that the data is of type int16 and in the form Buffer Rows x Total Elements On Transducer x Frames because on the Verasonics, all of the transducer elements are used to receive for the RF data buffer)
apply_params.display_image_flag = 1;  % Set to 0 to not display the processed images or set to 1 to display the processed images
apply_params.display_caxis_limits = [-60 0]; % Vector that specifies the caxis limits (dB) to use (only used when display_image_flag = 1)


%% Parameter Removal %%
% Remove the display_caxis_limits parameter if it isn't used
if apply_params.display_image_flag == 0
    fields_1 = {'display_caxis_limits'};
    apply_params = rmfield(apply_params, fields_1);    
end

tic;
%% Apply ADMIRE Models %%
% Apply the ADMIRE models to a set of channel data
if strcmp(apply_params.processor_type, 'CPU')
    apply_ADMIRE_models_CPU(apply_params);
elseif strcmp(apply_params.processor_type, 'GPU')
    apply_ADMIRE_models_GPU(apply_params);
end
toc;    