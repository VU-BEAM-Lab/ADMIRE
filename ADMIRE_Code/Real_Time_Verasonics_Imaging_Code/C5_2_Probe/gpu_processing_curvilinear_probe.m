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


% Description of gpu_processing_curvilinear_probe.m:
% This function obtains the parameters that are necessary for ADMIRE, and
% it processes the Verasonics buffer that contains the RF channel data. In
% addition, it displays the image for each frame and calculates the frame
% rate (FPS) for imaging and data processing.


function gpu_processing_curvilinear_probe(RcvData)

    %{
    % Uncomment this section and comment the rest of the lines below in
    % order to just save one example dataset for calculating some of the
    % parameters that are required for generating the ADMIRE models
    Trans = evalin('base', 'Trans');
    Receive = evalin('base', 'Receive');
    TW = evalin('base', 'TW');
    TX = evalin('base', 'TX');
    P = evalin('base', 'P');
    Resource = evalin('base', 'Resource');
    dtheta = evalin('base', 'dtheta');
    save('Dataset_Curvilinear_Probe.mat', 'RcvData', 'Trans', 'Receive', ...
        'TW', 'TX', 'P', 'Resource', 'dtheta');
    %}

    % Declare persistent variables and figure handles
    persistent flag_1 GPU_fixed_params_h delays_h stft_window_h selected_freq_inds_h ...
        negative_freq_inds_h negative_freq_include_h y_include_mask_h num_observations_h ...
        observation_thread_stride_h num_predictors_h X_matrix_thread_stride_h ...
        X_matrix_h B_thread_stride_h dr_h dth_h i00_h i01_h i10_h i11_h idx_h ...
        GPU_adjustable_params_h axial_positions lateral_positions ADMIRE_handle ...
        ax1 flag_2 frameCount startTime;
    
    % Obtain persistent variables (only need to do this once because most of
    % variables are fixed during scanning)
    if isempty(flag_1)
        flag_1 = 1;
        load ADMIRE_Variables_Curvilinear_Probe_Verasonics_RF_Buffer_Data_Type.mat;
        GPU_fixed_params_h = GPU_fixed_params;
        delays_h = delays;
        stft_window_h = stft_window;
        selected_freq_inds_h = selected_freq_inds;
        negative_freq_inds_h = negative_freq_inds;
        negative_freq_include_h = negative_freq_include;
        y_include_mask_h = y_include_mask;
        num_observations_h = num_observations;
        observation_thread_stride_h = observation_thread_stride;
        num_predictors_h = num_predictors;
        X_matrix_thread_stride_h = X_matrix_thread_stride;
        X_matrix_h = X_matrix;
        B_thread_stride_h = B_thread_stride;
        dr_h = dr;
        dth_h = dth;
        i00_h = i00;
        i01_h = i01;
        i10_h = i10;
        i11_h = i11;
        idx_h = idx;
        axial_positions = ax;
        lateral_positions = la;
    end

    % Obtain and store the values of the adjustable parameters for ADMIRE
    alpha = evalin('base','alpha');
    lambda_scaling_factor = evalin('base','lambda_scaling_factor');
    max_iterations = evalin('base','max_iterations');
    tolerance = evalin('base','tolerance');
    GPU_adjustable_params_h = single([alpha, tolerance, max_iterations, ...
        lambda_scaling_factor]);

    % Obtain the normalized, log-compressed, and scan-converted envelope of 
    % the data that is processed with ADMIRE
    [envelope] = ADMIRE_GPU_curvilinear_probe_Verasonics_RF_buffer_data_type(GPU_fixed_params_h, ...
        delays_h, stft_window_h, selected_freq_inds_h, negative_freq_inds_h, ...
        negative_freq_include_h, y_include_mask_h, num_observations_h, ...
        observation_thread_stride_h, num_predictors_h, X_matrix_thread_stride_h, ...
        X_matrix_h, B_thread_stride_h, dr_h, dth_h, i00_h, i01_h, i10_h, i11_h, ...
        idx_h, GPU_adjustable_params_h, RcvData);
    
    % Create the ADMIRE figure handle
    if isempty(ADMIRE_handle) || ~ishandle(ADMIRE_handle)
        figure(2);
        ax1 = gca;
        colormap gray;
        ADMIRE_handle = imagesc(lateral_positions, axial_positions, zeros(length(axial_positions), ...
            length(lateral_positions)), [-60 0]);
        axis image;
        colorbar;
    end

    % Set the normalized and log-compressed envelope data in order to display
    % it
    set(ADMIRE_handle, 'CData', envelope);

    % Create the x-axis and y-axis lables along with the title
    if isempty(flag_2)
        flag_2 = 1;
        xlabel(ax1, 'Lateral Position (cm)');
        ylabel(ax1, 'Depth (cm)');
        title(ax1, 'ADMIRE');
    end

    % Calculate the frame rate (FPS) for imaging and processing
    if isempty(frameCount)
        frameCount = 0;
        startTime = clock;
    else
        frameCount = frameCount + 1;
        if rem(frameCount, 100) == 0 
            RunningTime = etime(clock, startTime);
            frameRate = frameCount / RunningTime;
            disp(['Frame Rate (FPS): ', num2str(frameRate)]);
            frameCount = 0;
            startTime = clock;
        end
    end

end