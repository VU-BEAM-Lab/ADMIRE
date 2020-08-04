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


% Description of format_ADMIRE_models_for_GPU.m:
% This function formats the ADMIRE models in order for them to be applied
% on the GPU


function format_ADMIRE_models_for_GPU(params)

    %% Specify The Parameters That Are Required To Run ADMIRE On A GPU %%
    t0 = params.t0;                                    % Verasonics t0 sample index: Any depth indices less than t0 will be removed (set t0 to 1 in cases such as when data was not collected with a Verasonics system or when the t0 is already accounted for in the data)
    c = params.c;                                      % Speed of sound (m/s)       
    fs = params.fs;                                    % Sampling frequency (Hz)
    num_depths = params.num_depths;                    % Number of depth samples in the channel data
    num_elements = params.num_elements;                % Number of receive elements used to obtain one beam
    total_elements_on_probe = params.total_elements_on_probe; % Total number of elements on the transducer array
    num_beams = params.num_beams;                      % Number of beams
    stft_num_windows = params.stft_num_windows;        % Number of STFT windows for one beam
    stft_length = params.stft_window_length;           % STFT window length without zero-padding
    stft_window_shift = params.stft_window_shift;      % Number of depth samples to shift by when moving to the next STFT window
    num_selected_freqs = length(params.selected_freqs); % Number of selected frequencies within one STFT window to perform ADMIRE on
    stft_num_zeros = params.zero_padded_stft_window_length - params.stft_window_length; % Number of zeros that are used for STFT window zero padding
    max_windows_per_set = params.max_windows_per_set;    % Number of windows to group together in one set for STFT calculation on the GPU 
    num_corresponding_negative_freqs = num_selected_freqs - sum(params.selected_freqs == 0);  % Number of corresponding negative frequencies (The conjugates of the reconstructed ADMIRE signals for the positive frequencies are stored into the corresponding negative frequencies. A frequency of 0 Hz doesn't have a corresponding negative frequency.)
    num_fits = stft_num_windows .* num_selected_freqs .* num_beams;  % Number of model fits that are performed
    stft_window = window(params.stft_windowing_function, stft_length); % Windowing function used for the STFT
    selected_freq_inds = params.selected_freq_inds - 1;  % Indices of the selected frequencies within one STFT window (1 is subtracted to account for zero-based indexing on the GPU)
    negative_freq_include = params.selected_freqs ~= 0;  % This is a vector that determines which of the selected frequencies have a corresponding negative frequency (0 means it doesn't have a corresponding negative frequency, and 1 means that it does have a corresponding negative frequency)
    alpha = params.alpha;                                % The alpha that is used for elastic-net regularization
    lambda_scaling_factor = params.lambda_scaling_factor;    % Scaling factor that is used in the calculation of lambda, which is used in elastic-net regularization
    max_iterations = params.max_iterations;             % The maximum number of iterations of cyclic coordinate descent to perform
    tolerance = params.tolerance;                       % If the maximum coefficient change between iterations of cyclic coordinate descent falls below this number before the maximum number of iterations has been reached, then cyclic coordinate descent stops
    depths = params.depths;                              % This is a vector that contains the depth values (m) for the data
    if strcmp(params.data_type, 'Verasonics RF Buffer')
        num_buffer_rows = params.num_buffer_rows;        % The number of rows per frame in the Verasonics RF buffer: Not used when data_type = 'Reshaped'
    end
    channel_data_output_flag = params.channel_data_output_flag;  % Set to 0 to only have the envelope data outputted when the ADMIRE models are applied or set to 1 to have both the envelope data and channel data outputted when the ADMIRE models are applied
    
    % Determine the indices of the negative frequencies that correspond to
    % the positive selected frequencies
    negative_freq_inds = zeros(1, num_corresponding_negative_freqs);  % Indices of corresponding negative frequencies (1 is subtracted to account for zero-based indexing on the GPU) 
    if mod(params.zero_padded_stft_window_length, 2) == 1
        freqs = ([(-(params.zero_padded_stft_window_length - 1) ./ 2):((params.zero_padded_stft_window_length - 1) ./ 2)] ./ params.zero_padded_stft_window_length) .* params.fs;
        negative_half = freqs(1:((params.zero_padded_stft_window_length - 1) ./ 2));
        rearranged_freqs = [freqs((((params.zero_padded_stft_window_length - 1) ./ 2) + 1:end)) negative_half];
        count = 1;
        for ii = 1:num_selected_freqs
            if params.selected_freqs(ii) ~= 0
                ind = find(-params.selected_freqs(ii) == rearranged_freqs);
                negative_freq_inds(count) = ind - 1;
                count = count + 1;
            end
        end
    elseif mod(params.zero_padded_stft_window_length, 2) == 0
        freqs = ([(-params.zero_padded_stft_window_length ./ 2):((params.zero_padded_stft_window_length ./ 2) - 1)] ./ params.zero_padded_stft_window_length) .* params.fs;
        negative_half = freqs(1:(params.zero_padded_stft_window_length ./ 2));
        rearranged_freqs = [freqs(((params.zero_padded_stft_window_length ./ 2) + 1):end) negative_half];
        count = 1;
        for ii = 1:num_selected_freqs
            if params.selected_freqs(ii) ~= 0
                ind = find(-params.selected_freqs(ii) == rearranged_freqs);
                negative_freq_inds(count) = ind - 1;
                count = count + 1;
            end
        end
    end
    
    % Obtain the depth offset if ADMIRE is applied to a range that is
    % smaller than the entire depth range
    filename = ['STFT_window_number_1_frequency_number_1.mat'];
    load(fullfile(params.models_save_path, filename));
    start_depth_offset = params.stft_start_depth_inds(params.stft_window_inds(1));  % Depth offset in samples to use when ADMIRE is applied to a range that is smaller than the entire depth range
    
    
    %% Organize Model Matrices And Obtain Start Indices %%
    % Loop through all of the models in order to organize the model
    % matrices and obtain the start indices for each model
    X_matrix = [];   % This contains all of the models across all of the STFT windows as one column vector
    X_matrix_thread_stride = [0];  % This contains the indices for where each model begins in the X_matrix_h column vector
    num_predictors = [];   % This contains the number of predictors for each model
    num_observations = []; % This contains the number of observations for each model
    y_include_mask = [];   % This contains the aperture growth mask for each model
    
    for window_number = 1:params.stft_num_windows
        for freq_number = 1:num_selected_freqs
            filename = ['STFT_window_number_' num2str(window_number) '_frequency_number_' num2str(freq_number) '.mat'];
            load(fullfile(params.models_save_path, filename));
            X_matrix = [X_matrix; combined_model(:)];
            X_matrix_thread_stride = [X_matrix_thread_stride; length(X_matrix(:))];
            num_predictors = [num_predictors; size(combined_model, 2)];
            num_observations = [num_observations; size(combined_model, 1)];
            model_aperture_mask = repmat(params.aperture_growth_mask, [1, 2])';  % This is replicated because in the GPU code, the real and complex components of the STFT of the channel data are separate, so the same mask is applied to both components for a given model
            y_include_mask = [y_include_mask; model_aperture_mask];
        end
    end
    
    % Remove the last index value because it is not needed
    X_matrix_thread_stride = X_matrix_thread_stride(1:end - 1);
    
    % Replicate the indices because each beam uses the same indices
    X_matrix_thread_stride = repmat(X_matrix_thread_stride, [1 num_beams]);
    
    % Turn the matrix of indices into a column vector
    X_matrix_thread_stride = X_matrix_thread_stride(:);
    
    % Obtain the total number of values across all of the model matrices
    total_num_X_matrix_values = length(X_matrix);
    
    % Replicate the aperture growth masks because all of the beams use the
    % same masks
    y_include_mask = repmat(y_include_mask, [1 num_beams]);
    
    % Turn the matrix of aperture growth masks into a column vector
    y_include_mask = y_include_mask(:);
    
    % Replicate the model predictor counts because the same counts are used
    % by all of the beams
    num_predictors = repmat(num_predictors, [1 num_beams]);
    
    % Turn the matrix of predictor counts into a column vector
    num_predictors = num_predictors(:);
    
    % Replicate the model observation counts because the same counts are
    % used by all of the beams
    num_observations = repmat(num_observations, [1 num_beams]);
    
    % Turn the matrix of predictor counts into a column vector
    num_observations = num_observations(:);
    
    observation_thread_stride = [0];  % This contains the indices for where each set of observations begins
    B_thread_stride = [0];   % This contains the indices for where each set of predictor coefficients begins
    sum_observation = 0;   
    sum_B = 0;
    
    % Determine the start indices for the observation sets and the
    % predictor coefficient sets for the models
    for beam = 1:num_beams
        count = 1;
        for window_number = 1:params.stft_num_windows
            for freq_number = 1:num_selected_freqs
                sum_B = sum_B + num_predictors(count);
                sum_observation = sum_observation + num_observations(count);
                observation_thread_stride = [observation_thread_stride; sum_observation];
                B_thread_stride = [B_thread_stride; sum_B];
                count = count + 1;
            end
        end
    end
    
    % Remove the last index value because it is not needed
    observation_thread_stride = observation_thread_stride(1:end - 1);
    
    % Remove the last index value because it is not needed
    B_thread_stride = B_thread_stride(1:end - 1);
    
    % Obtain the total number of observations and the total number of
    % predictor coefficients across all of the models
    total_num_cropped_y_observations = sum_observation;  % Number of observations across all of the model fits once aperture growth is taken into account
    total_num_B_values = sum_B;   % Number of predictor coefficients across all of the models
    
    % Convert some of the arrays to single precision
    stft_window = single(stft_window);
    selected_freq_inds = single(selected_freq_inds);
    negative_freq_inds = single(negative_freq_inds);
    negative_freq_include = single(negative_freq_include);
    depths = single(depths);
    y_include_mask = single(y_include_mask);
    X_matrix = single(X_matrix);
    
    
    %% Obtain GPU Inputs That Are Required For A Linear Probe %%
    if strcmp(params.probe_type, 'Linear')
        % Calculate the element positions for one beam
        elem_pos_x = [0:num_elements - 1] .* params.probe_pitch;   
        elem_pos_x = elem_pos_x - mean(elem_pos_x);  % Lateral positions of the elements (m)
        
        % Calculate the delays that are applied to the channel data
        elem_pos_x_matrix = repmat(elem_pos_x, [num_depths, 1]);
        depths_matrix = repmat(depths', [1, num_elements]);
        delays = ((1 ./ c) .* sqrt((elem_pos_x_matrix .^ 2) + (depths_matrix .^ 2))) + (depths_matrix ./ c);
        delays = delays .* fs;
        delays = repmat(delays, [1, 1, num_beams]);  % Delays for the channel data (sample shifts)
        
        % Calculate the beam positions
        beam_pos_x = [0:num_beams - 1] .* params.probe_pitch;
        beam_pos_x = beam_pos_x - mean(beam_pos_x); % Lateral positions of the beams (m)  
        
        % Convert the above matrix and array to single precision
        delays = single(delays);
        beam_pos_x = single(beam_pos_x);
    end
    
     
    %% Obtain GPU Inputs That Are Required For A Curvilinear Probe %%
    if strcmp(params.probe_type, 'Curvilinear')
        % Obtain these parameters from the parameters structure
        dtheta = params.dtheta;           % Angle increment between beams in (rad): Not used when probe_type = 'Linear'
        radius = params.probe_radius;     % Curved transducer array radius (m): Not used when probe_type = 'Linear'
        
        % Calculate the element positions for one beam
        e = [0:total_elements_on_probe - 1] - mean([0:total_elements_on_probe - 1]);
        elem_pos_x = radius .* sin(dtheta .* e);  % Lateral positions of all of the elements on the transducer (m)
        elem_pos_z = radius .* cos(dtheta .* e);  % Axial positions of all of the elements on the transducer (m)
        
        % Calculate the delays that are applied to the channel data
        center_depths = params.depths;
        beam = [0:num_beams - 1] - mean([0:num_beams - 1]);
        delays = zeros(num_depths, num_elements, num_beams);
        for beam_ind = 1:num_beams
            bx = (center_depths + radius) .* sin(dtheta .* beam(beam_ind));
            bz = (center_depths + radius) .* cos(dtheta .* beam(beam_ind));
            bx = repmat(bx', [1, num_elements]);
            bz = repmat(bz', [1, num_elements]);
            elem_pos_x_beam = elem_pos_x(beam_ind:(beam_ind + num_elements - 1));
            elem_pos_z_beam = elem_pos_z(beam_ind:(beam_ind + num_elements - 1));
            elem_pos_x_beam = repmat(elem_pos_x_beam, [num_depths, 1]);
            elem_pos_z_beam = repmat(elem_pos_z_beam, [num_depths, 1]);
            delays(:, :, beam_ind) = ((1 ./ c) .* sqrt(((elem_pos_x_beam - bx) .^ 2) ...
                + ((elem_pos_z_beam - bz) .^ 2))) + ((sqrt((bx .^ 2) + (bz .^ 2)) - radius) ./ c);
        end
        
        % Convert the time delays to sample shifts
        delays = delays .* fs;  % Delays for the channel data (sample shifts)
        
        % Convert the delays matrix to single precision
        delays = single(delays);  
            
        % Obtain the parameters that are required for scan conversion 
        % This matrix is just a matrix of zeros because the GPU code
        % actually applies the scan conversion to the image data using the
        % parameters that are returned by the sconvert function for an
        % image of this size
        image_data_zeros = zeros(num_depths, num_beams);  
        
        % Calculate the scan sector angle in degrees that each beam corresponds to
        sector = ([0:num_beams - 1] - mean([0:num_beams - 1])) .* dtheta .* (180 ./ pi);
        
        % Determine the minimum scan sector angle in degrees
        min_phi = sector(1);
        
        % Determine the angle span in degrees of the scan
        span_phi = sector(end) - sector(1);
        
        % Define the distance to the radial center of the probe in cm
        apex = -radius .* 100;
        
        % Define the variable that determines whether to do nearest
        % neighbor interpolation or bilinear interpolation for scan
        % conversion (1 = nearest neighbor and 2 = bilinear interpolation)
        dsfactor = 2;
        
        % Define the array is used to determine the scan conversion grid
        vargin = [0 0 5E-5 0 0 5E-5];  % [ax_min, ax_max, ax_inc, lat_min, lat_max, lat_inc] (meters)
        
        % Obtain the scan conversion parameters using the sconvert and
        % scmap functions
        [out, ax, la, ~, dr, dth, i00, i01, i10, i11, idx] = sconvert(image_data_zeros, ...
            'sector', min_phi, span_phi, apex, dsfactor, fs, c, vargin);
        
        % Subtract 1 from the arrays that contain indices in order to 
        % account for zero-based indexing on the GPU
        i00 = i00 - 1;    % These arrays contain indices that are used to perform scan conversion on the GPU
        i01 = i01 - 1;
        i10 = i10 - 1;
        i11 = i11 - 1;
        idx = idx - 1;
        scan_conversion_parameters_length = length(i00);  % This gives the length of each of these arrays because all of them are the same length
        
        % Obtain the dimensions of the scan converted image
        scan_converted_num_axial_positions = size(out, 1);   % Number of depths in the scan converted image
        scan_converted_num_lateral_positions = size(out, 2);  % Number of lateral positions in the scan converted image 
        
        % Convert the above arrays to single precision
        ax = single(ax);     % This contains the depth values (cm) for the scan converted image
        la = single(la);     % This contains the lateral position values (cm) for the scan converted image
        i00 = single(i00);
        i01 = single(i01);
        i10 = single(i10);
        i11 = single(i11);
        idx = single(idx);
        dr = single(dr);     % This array is used for scan conversion
        dth = single(dth);   % This array is used for scan conversion   
    end
    
    
    %% Save Out GPU Inputs %%
    % Save out the GPU inputs for the linear probe cases
    if strcmp(params.probe_type, 'Linear')
        % Save out the GPU inputs for the case where the data type is
        % Verasonics RF Buffer
        if strcmp(params.data_type, 'Verasonics RF Buffer')
            GPU_fixed_params = [t0, num_buffer_rows, num_depths, total_elements_on_probe, ...
                num_elements, num_beams, start_depth_offset, stft_num_zeros, ...
                stft_num_windows, stft_window_shift, stft_length, max_windows_per_set, ...
                num_selected_freqs, num_corresponding_negative_freqs, num_fits, ...
                total_num_cropped_y_observations, total_num_X_matrix_values, ...
                total_num_B_values, channel_data_output_flag]; % This array contains the parameters that are required to run ADMIRE on a GPU (these parameters are meant to be fixed)
            GPU_adjustable_params = [alpha, tolerance, max_iterations, lambda_scaling_factor];
            
            % Convert the GPU_adjustable_params array to single precision
            GPU_adjustable_params = single(GPU_adjustable_params);  % This array contains the parameters that are required to run ADMIRE on a GPU (these parameters can be modified, such as changing alpha to see its effect on ADMIRE processing)
            
            % Obtain the names of the parameters in the arrays
            fixed_parameter_names = {'t0', 'num_buffer_rows', 'num_depths', 'total_elements_on_probe', ...
                'num_elements', 'num_beams', 'start_depth_offset', 'stft_num_zeros', ...
                'stft_num_windows', 'stft_window_shift', 'stft_length', 'max_windows_per_set', ...
                'num_selected_freqs', 'num_corresponding_negative_freqs', ...
                'num_fits', 'total_num_cropped_y_observations', 'total_num_X_matrix_values', ...
                'total_num_B_values', 'channel_data_output_flag'};
            adjustable_parameter_names = {'alpha', 'tolerance', 'max_iterations', ...
                'lambda_scaling_factor'};
            
            % Save to the directory provided by params.models_save_path
            filename = ['ADMIRE_Variables_Linear_Probe_Verasonics_RF_Buffer_Data_Type.mat'];
            save(fullfile(params.models_save_path, filename), '-v7.3', 'GPU_fixed_params', ...
                'GPU_adjustable_params', 'delays', 'depths', 'beam_pos_x', ...
                'stft_window', 'selected_freq_inds', 'negative_freq_inds', ...
                'negative_freq_include', 'y_include_mask', 'num_observations', ...
                'observation_thread_stride', 'num_predictors', 'X_matrix_thread_stride', ...
                'X_matrix', 'B_thread_stride', 'fixed_parameter_names', 'adjustable_parameter_names');
        elseif strcmp(params.data_type, 'Reshaped') 
            % Save out the GPU inputs for the case where the data type is
            % Reshaped
            GPU_fixed_params = [t0, num_depths, num_elements, num_beams, start_depth_offset, ...
                stft_num_zeros, stft_num_windows, stft_window_shift, stft_length, ...
                max_windows_per_set, num_selected_freqs, num_corresponding_negative_freqs, ...
                num_fits, total_num_cropped_y_observations, total_num_X_matrix_values, ...
                total_num_B_values, channel_data_output_flag]; 
            GPU_adjustable_params = [alpha, tolerance, max_iterations, lambda_scaling_factor];
            
            % Convert the GPU_adjustable_params arrays to single precision
            GPU_adjustable_params = single(GPU_adjustable_params);
            
            % Obtain the names of the parameters in the arrays
            fixed_parameter_names = {'t0', 'num_depths', 'num_elements', 'num_beams', ...
                'start_depth_offset', 'stft_num_zeros', 'stft_num_windows', ...
                'stft_window_shift', 'stft_length', 'max_windows_per_set', ...
                'num_selected_freqs','num_corresponding_negative_freqs', 'num_fits', ...
                'total_num_cropped_y_observations', 'total_num_X_matrix_values', ...
                'total_num_B_values', 'channel_data_output_flag'};
            adjustable_parameter_names = {'alpha', 'tolerance', 'max_iterations', ...
                'lambda_scaling_factor'};
            
            % Save to the directory provided by params.models_save_path
            filename = ['ADMIRE_Variables_Linear_Probe_Reshaped_Data_Type.mat'];
            save(fullfile(params.models_save_path, filename), '-v7.3', 'GPU_fixed_params', ...
                'GPU_adjustable_params', 'delays', 'depths', 'beam_pos_x', ...
                'stft_window', 'selected_freq_inds', 'negative_freq_inds', ...
                'negative_freq_include', 'y_include_mask', 'num_observations', ...
                'observation_thread_stride', 'num_predictors', 'X_matrix_thread_stride', ...
                'X_matrix', 'B_thread_stride', 'fixed_parameter_names', 'adjustable_parameter_names');            
        end
    end
    
    % Save out the GPU inputs for the curvilinear probe cases
    if strcmp(params.probe_type, 'Curvilinear')
        % Save out the GPU inputs for the case where the data type is
        % Verasonics RF Buffer
        if strcmp(params.data_type, 'Verasonics RF Buffer')
            GPU_fixed_params = [t0, num_buffer_rows, num_depths, total_elements_on_probe, ...
                num_elements, num_beams, start_depth_offset, stft_num_zeros, ...
                stft_num_windows, stft_window_shift, stft_length, max_windows_per_set, ...
                num_selected_freqs, num_corresponding_negative_freqs, num_fits, ...
                total_num_cropped_y_observations, total_num_X_matrix_values, ...
                total_num_B_values, scan_conversion_parameters_length, scan_converted_num_axial_positions, ...
                scan_converted_num_lateral_positions, channel_data_output_flag];
            GPU_adjustable_params = [alpha, tolerance, max_iterations, lambda_scaling_factor];            
            
            % Convert the GPU_adjustable_params array to single precision
            GPU_adjustable_params = single(GPU_adjustable_params);
            
            % Obtain the names of the parameters in the array
            fixed_parameter_names = {'t0', 'num_buffer_rows', 'num_depths', ...
                'total_elements_on_probe', 'num_elements', 'num_beams', 'start_depth_offset', ...
                'stft_num_zeros', 'stft_num_windows', 'stft_window_shift', ...
                'stft_length', 'max_windows_per_set', 'num_selected_freqs', ...
                'num_corresponding_negative_freqs', 'num_fits', 'total_num_cropped_y_observations', ...
                'total_num_X_matrix_values', 'total_num_B_values', 'scan_conversion_parameters_length', ...
                'scan_converted_num_axial_positions', 'scan_converted_num_lateral_positions', ...
                'channel_data_output_flag'};
            adjustable_parameter_names = {'alpha', 'tolerance', 'max_iterations', ...
                'lambda_scaling_factor'};            
            
            % Save to the directory provided by params.models_save_path
            filename = ['ADMIRE_Variables_Curvilinear_Probe_Verasonics_RF_Buffer_Data_Type.mat'];
            save(fullfile(params.models_save_path, filename), '-v7.3', 'GPU_fixed_params', ...
                'GPU_adjustable_params', 'delays', 'stft_window', 'selected_freq_inds', ...
                'negative_freq_inds', 'negative_freq_include', 'y_include_mask', ...
                'num_observations', 'observation_thread_stride', 'num_predictors', ...
                'X_matrix_thread_stride', 'X_matrix', 'B_thread_stride', 'dr', ...
                'dth', 'i00', 'i01', 'i10', 'i11', 'idx', 'ax', 'la', 'fixed_parameter_names', ...
                'adjustable_parameter_names');
        elseif strcmp(params.data_type, 'Reshaped')
            % Save out the GPU inputs for the case where the data type is
            % Reshaped
            GPU_fixed_params = [t0, num_depths, num_elements, num_beams, start_depth_offset, ...
                stft_num_zeros, stft_num_windows, stft_window_shift, stft_length, ...
                max_windows_per_set, num_selected_freqs, num_corresponding_negative_freqs, ...
                num_fits, total_num_cropped_y_observations, total_num_X_matrix_values, ...
                total_num_B_values, scan_conversion_parameters_length, scan_converted_num_axial_positions, ...
                scan_converted_num_lateral_positions, channel_data_output_flag];
            GPU_adjustable_params = [alpha, tolerance, max_iterations, lambda_scaling_factor];             
            
            % Convert the GPU_adjustable_params array to single precision
            GPU_adjustable_params = single(GPU_adjustable_params);
            
            % Obtain the names of the parameters in the array
            fixed_parameter_names = {'t0', 'num_depths', 'num_elements', 'num_beams', ...
                'start_depth_offset', 'stft_num_zeros', 'stft_num_windows', ...
                'stft_window_shift', 'stft_length', 'max_windows_per_set', ...
                'num_selected_freqs', 'num_corresponding_negative_freqs', ...
                'num_fits', 'total_num_cropped_y_observations', 'total_num_X_matrix_values', ...
                'total_num_B_values', 'scan_conversion_parameters_length', ...
                'scan_converted_num_axial_positions', 'scan_converted_num_lateral_positions', ...
                'channel_data_output_flag'};
            adjustable_parameter_names = {'alpha', 'tolerance', 'max_iterations', ...
                'lambda_scaling_factor'};              
            
            % Save to the directory provided by params.models_save_path
            filename = ['ADMIRE_Variables_Curvilinear_Probe_Reshaped_Data_Type.mat'];
            save(fullfile(params.models_save_path, filename), '-v7.3', 'GPU_fixed_params', ...
                'GPU_adjustable_params', 'delays', 'stft_window', 'selected_freq_inds', ...
                'negative_freq_inds', 'negative_freq_include', 'y_include_mask', ...
                'num_observations', 'observation_thread_stride', 'num_predictors', ...
                'X_matrix_thread_stride', 'X_matrix', 'B_thread_stride', 'dr', ...
                'dth', 'i00', 'i01', 'i10', 'i11', 'idx', 'ax', 'la', 'fixed_parameter_names', ...
                'adjustable_parameter_names');     
        end
    end
    
end