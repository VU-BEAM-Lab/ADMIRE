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


% Description of generate_models_for_stft_window.m:
% This function generates the models for the selected frequencies in each
% STFT window


function generate_models_for_stft_window(window_id, params)

    %% Parameter Calculation %%
    % Define the number of selected frequencies within one STFT window
    num_selected_freqs = length(params.selected_freqs);

    % Look up the wavenumber calibration values 
    wavenumber_cal_values = wavenumber_calibration_value_lookup(params);
    
    % Check to make sure that the number of selected frequencies is less
    % than or equal to the number of wavenumber calibration values that are
    % returned
    if num_selected_freqs > length(wavenumber_cal_values)
        error('The number of selected frequencies exceeds the number of wavenumber calibration values. Either use a smaller number of selected frequencies or add more wavenumber calibration values for the corresponding probe and frequency case in the wavenumber_calibration_value_lookup function.');
    end

    % Calculate the calibrated wavenumber values
    k_calibrated = params.k(1:num_selected_freqs) .* wavenumber_cal_values(1:num_selected_freqs);
    
    % Store the calibrate wavenumber values into the parameters structure
    params.k_calibrated = k_calibrated;        % Calibrated wavenumber values corresponding to the selected frequencies (rad/m)

    % Determine the depth samples that are present in the current STFT window
    window_ind = params.stft_window_inds(window_id);
    start_depth_ind = params.stft_start_depth_inds(window_ind);
    end_depth_ind = start_depth_ind + params.stft_window_length - 1;
    depths_in_window = params.depths(start_depth_ind:end_depth_ind);
    
    % Store the depth samples that are present in the current STFT window
    params.depths_in_window = depths_in_window;    % Depths for the STFT window (m)

    % Determine the center depth of the current STFT window
    z_center_stft_window = mean(params.depths_in_window);

    % Set the center lateral position of the current STFT window to 0 m
    x_center_stft_window = 0;
    
    
    %% Aperture Growth Mask Calculation %%
    % Calculate the aperture growth mask to apply to the data (aperture
    % growth is done based off of the center depth of the STFT window)
    aperture_growth_mask = zeros(1, params.num_elements);
    if params.aperture_growth_flag == 1
        num_elements_stft_window = ceil(z_center_stft_window ./ params.probe_pitch ./ params.F_number);
        num_elements_stft_window = 2 .* ceil(num_elements_stft_window ./ 2);
        num_elements_stft_window = min(max(num_elements_stft_window, params.min_num_elements), params.num_elements);
        element_indices = ceil(params.num_elements ./ 2 + ((1 - num_elements_stft_window ./ 2):(num_elements_stft_window ./ 2)));
        aperture_growth_mask(element_indices) = 1;      
    else 
        num_elements_stft_window = params.num_elements;
        aperture_growth_mask(1:params.num_elements) = 1;
    end
    
    % Store the number of elements for the STFT window into the parameters structure
    params.num_elements_stft_window = num_elements_stft_window;    % Number of elements for the STFT window after taking aperture growth into account
    
    % Store the aperture growth mask into the parameters structure
    params.aperture_growth_mask = aperture_growth_mask;     % Aperture growth mask for the STFT window
   

    %% Element Lateral Position Calculation %%
    % Calculate the element lateral positions for the STFT window
    elem_pos_x = [1:params.num_elements_stft_window] - mean([1:params.num_elements_stft_window]);
    elem_pos_x = elem_pos_x .* params.probe_pitch;
    
    % Store the element lateral positions for the STFT window
    params.elem_pos_x = elem_pos_x;         % Element lateral positions for the STFT window 
    

    %% Model Generation For Each Frequency %%
    % Loop through the frequencies and generate the model for each one
    for freq_num = 1:num_selected_freqs
        % Calculate the wavelength for the frequency
        wavelength = (2 .* pi) ./ params.k_calibrated(freq_num);
        
        % Determine the model sampling by calculating the predictor signal for
        % the predictor in the center of the STFT window
        model_x_position_space = [x_center_stft_window, 1, x_center_stft_window];  % Vector defining the min, spacing, and max for the possible lateral positions in the model space
        model_z_position_space = [z_center_stft_window, 1, z_center_stft_window];  % Vector defining the min, spacing, and max for the possible depth positions in the model space
        model_distance_offset_space = [0, 1, 0];                                   % Vector defining the min, spacing, and max for the possible depth offsets in the model space (used in the calculation of tau_n0)
        model_space_matrix = [model_x_position_space; model_z_position_space; model_distance_offset_space];  % Matrix defining the model space
        
        % Generate the model using the defined model space
        model_for_frequency = generate_model_for_frequency(freq_num, model_space_matrix, z_center_stft_window, params);
        
        % Calculate the lateral resolution based on the bandwidth of the
        % predcitor signal in the center of the STFT window
        fft_model_for_frequency = abs(fft(abs(model_for_frequency), 2 .^ (2 .* floor(log2(params.num_elements_stft_window)))));
        fft_model_for_frequency = fft_model_for_frequency ./ max(fft_model_for_frequency);
        fwhm_width = sum(fft_model_for_frequency > 0.5);
        smpl_Fs = (1 ./ (params.probe_pitch)) ./ length(fft_model_for_frequency);
        res_lat = (fwhm_width .* smpl_Fs) .* wavelength .* z_center_stft_window;
     
        % Calculate the axial resolution based off of the lateral
        % resolution
        res_axl = 2 .* res_lat;
        
        % Obtain the scaling factors that are used to calculate the
        % possible lateral positions of the ROI model space
        ROI_scale_1_x = params.ROI_model_x_position_scaling_factors(1);
        ROI_scale_2_x = params.ROI_model_x_position_scaling_factors(2);
        ROI_scale_3_x = params.ROI_model_x_position_scaling_factors(3);
        
        % Calculate the min, spacing, and max for the possible lateral positions of 
        % the ROI model space
        ROI_min_x = ROI_scale_1_x .* res_lat;
        ROI_spacing_x = ROI_scale_2_x .* res_lat;
        ROI_max_x = ROI_scale_3_x .* res_lat;
        
        % Obtain the scaling factors that are used to calculate the
        % possible depth positions of the ROI model space
        ROI_scale_1_z = params.ROI_model_z_position_scaling_factors(1);
        ROI_scale_2_z = params.ROI_model_z_position_scaling_factors(2);
        ROI_scale_3_z = params.ROI_model_z_position_scaling_factors(3);
        
        % Calculate the min, spacing, and max for the possible depth
        % positions of the ROI model space
        ROI_min_z = z_center_stft_window + (ROI_scale_1_z .* res_axl);
        ROI_spacing_z = ROI_scale_2_z .* res_axl;
        ROI_max_z = z_center_stft_window + (ROI_scale_3_z .* res_axl);
        
        % Obtain the scaling factor and the constants that are used to
        % calculate the possible distance offsets for the ROI model space
        ROI_scale_1_distance_offset = params.ROI_model_distance_offset_scaling_factor;
        ROI_constant_1_distance_offset = params.ROI_model_distance_offset_constants(1);
        ROI_constant_2_distance_offset = params.ROI_model_distance_offset_constants(2);
        
        % Define the min and max and calculate the spacing for the possible
        % distance offsets for the ROI model space
        ROI_min_distance_offset = ROI_constant_1_distance_offset;
        ROI_spacing_distance_offset = (ROI_scale_1_distance_offset .* 2 .* pi) ./ params.k_calibrated(freq_num);
        ROI_max_distance_offset = ROI_constant_2_distance_offset;
        
        % Define the model space for the ROI (Region of Interest)
        % predictors (these variables can be hard-coded by the user, but it is not recommended)
        ROI_model_x_position_space = [ROI_min_x, ROI_spacing_x, ROI_max_x];
        ROI_model_z_position_space = [ROI_min_z, ROI_spacing_z, ROI_max_z];
        ROI_model_distance_offset_space = [ROI_min_distance_offset, ROI_spacing_distance_offset, ROI_max_distance_offset];
        ROI_model_space_matrix = [ROI_model_x_position_space; ROI_model_z_position_space; ROI_model_distance_offset_space];
       
        % Generate the ROI model using the defined model space
        [ROI_model_for_frequency, ROI_predictor_x_z_distance_offset_matrix] = generate_model_for_frequency(freq_num, ROI_model_space_matrix, z_center_stft_window, params);
        
        % Obtain the predictors that are within an ellipsoidal acceptance
        % region (this is the actual ROI, hence ROI model)
        ROI_predictor_x_positions = ROI_predictor_x_z_distance_offset_matrix(:, 1);
        ROI_predictor_z_positions = ROI_predictor_x_z_distance_offset_matrix(:, 2);
        ROI_x_component_ellipsoid = (ROI_predictor_x_positions - x_center_stft_window) ./ (res_lat + params.ellipsoid_constant_1);
        ROI_z_component_ellipsoid = (ROI_predictor_z_positions - z_center_stft_window) ./ (res_axl + params.ellipsoid_constant_1);
        ROI_ellipsoidal_acceptance_region_inds = ((ROI_x_component_ellipsoid .^ 2) + (ROI_z_component_ellipsoid .^ 2)) <= 1;
        ROI_model_for_frequency = ROI_model_for_frequency(:, ROI_ellipsoidal_acceptance_region_inds);
        ROI_predictor_x_z_distance_offset_matrix = ROI_predictor_x_z_distance_offset_matrix(ROI_ellipsoidal_acceptance_region_inds, :);
        
        % Obtain the scaling factors that are used to calculate the
        % possible lateral positions of the outer model space
        outer_scale_1_x = params.outer_model_x_position_scaling_factors(1);
        outer_scale_2_x = params.outer_model_x_position_scaling_factors(2);
        outer_scale_3_x = params.outer_model_x_position_scaling_factors(3);
        
        % Calculate the min, spacing, and max for the possible lateral
        % positions of the outer model space
        lateral_limit = (params.probe_pitch .* (params.num_elements_stft_window ./ 2)) + params.lateral_limit_offset;
        outer_min_x = outer_scale_1_x .* lateral_limit;
        outer_spacing_x = outer_scale_2_x .* res_lat;
        outer_max_x = outer_scale_3_x .* lateral_limit;
        
        % Obtain the scaling factors and constant that are used to
        % calculate the possible depth positions of the outer model space
        outer_scale_1_z = params.outer_model_z_position_scaling_factors(1);
        outer_scale_2_z = params.outer_model_z_position_scaling_factors(2);
        outer_scale_3_z = params.outer_model_z_position_scaling_factors(3);
        outer_scale_4_z = params.outer_model_z_position_scaling_factors(4);
        outer_scale_5_z = params.outer_model_z_position_scaling_factors(5);
        outer_scale_6_z = params.outer_model_z_position_scaling_factors(6);
        outer_constant_1_z = params.outer_model_z_position_constants(1);
        
        % Calculate the min, spacing, and max for the possible depth
        % positions of the outer model space
        outer_min_z = (outer_scale_1_z .* z_center_stft_window) + ...
            (outer_scale_2_z .* res_axl) + (outer_constant_1_z);
        outer_spacing_z = outer_scale_3_z .* res_axl;
        outer_max_z = (outer_scale_4_z .* z_center_stft_window) + ...
            (outer_scale_5_z .* res_axl) + (outer_scale_6_z .* z_center_stft_window);
        
        % Obtain the scaling factor and constants that are used to
        % calculate the possible distance offsets for the outer model space
        outer_scale_1_distance_offset = params.outer_model_distance_offset_scaling_factor(1);
        outer_constant_1_distance_offset = params.outer_model_distance_offset_constants(1);
        outer_constant_2_distance_offset = params.outer_model_distance_offset_constants(2);
        
        % Define the min and max and calculate the spacing for the possible 
        % distance offsets for the outer model space
        outer_min_distance_offset = outer_constant_1_distance_offset;
        outer_spacing_distance_offset = (outer_scale_1_distance_offset .* 2 .* pi) ...
            ./ params.k_calibrated(freq_num);
        outer_max_distance_offset = outer_constant_2_distance_offset;
        
        % Define the model space for the predictors that are outside of the
        % ROI (these variables can be hard-coded, but it is not
        % recommended)
        outer_model_x_position_space = [outer_min_x, outer_spacing_x, outer_max_x];
        outer_model_z_position_space = [outer_min_z, outer_spacing_z, outer_max_z];
        outer_model_distance_offset_space = [outer_min_distance_offset, outer_spacing_distance_offset, outer_max_distance_offset];
        outer_model_space_matrix = [outer_model_x_position_space; outer_model_z_position_space; outer_model_distance_offset_space];
        
        % Generate the outer model using the defined model space
        [outer_model_for_frequency, outer_predictor_x_z_distance_offset_matrix] = generate_model_for_frequency(freq_num, outer_model_space_matrix, z_center_stft_window, params);
        
        % Obtain the predictors that are outside of an ellipsoidal acceptance
        % region (the ellipsoid defines the ROI, so this actually obtains
        % the predictors that are outside of the ROI, hence outer model)
        outer_predictor_x_positions = outer_predictor_x_z_distance_offset_matrix(:, 1);
        outer_predictor_z_positions = outer_predictor_x_z_distance_offset_matrix(:, 2);
        outer_x_component_ellipsoid = (outer_predictor_x_positions - x_center_stft_window) ./ (res_lat + params.ellipsoid_constant_2);
        outer_z_component_ellipsoid = (outer_predictor_z_positions - z_center_stft_window) ./ (res_axl + params.ellipsoid_constant_2);
        outer_ellipsoidal_acceptance_region_inds = ((outer_x_component_ellipsoid .^ 2) + (outer_z_component_ellipsoid .^ 2)) > 1;
        outer_model_for_frequency = outer_model_for_frequency(:, outer_ellipsoidal_acceptance_region_inds);
        outer_predictor_x_z_distance_offset_matrix = outer_predictor_x_z_distance_offset_matrix(outer_ellipsoidal_acceptance_region_inds, :);
        
        % Perform ICA on the ROI and outer models if the ICA_ADMIRE_flag =
        % 1 
        if params.ICA_ADMIRE_flag == 1
            ROI_W = ica(ROI_model_for_frequency);
            ROI_model_for_frequency = pinv(ROI_W);
            outer_W = ica(outer_model_for_frequency);
            outer_model_for_frequency = pinv(outer_W);
        end
        
        % Combine the ROI and outer models
        combined_model = [ROI_model_for_frequency outer_model_for_frequency];
        
        % Standardize the model predictors
        combined_model = combined_model ./ sqrt(sum(abs(combined_model) .^ 2));
        
        % Separate out the real and imaginary components of the model
        % matrix (this is done because the implementation of the coordinate 
        % descent optimization algorithm does not take complex data as an
        % input)
        combined_model = [[real(combined_model); imag(combined_model)] ...
            [-imag(combined_model); real(combined_model)]];
        
        % Obtain the indices for the ROI and outer models in the combined
        % model
        ROI_model_first_component_indices = [1:size(ROI_model_for_frequency, 2)];
        outer_model_first_component_indices = [1:size(outer_model_for_frequency, 2)];
        outer_model_first_component_indices = outer_model_first_component_indices ...
            + size(ROI_model_for_frequency, 2);
        ROI_model_second_component_indices = [1:size(ROI_model_for_frequency, 2)];
        ROI_model_second_component_indices = ROI_model_second_component_indices ...
            + outer_model_first_component_indices(end);
        outer_model_second_component_indices = [1:size(outer_model_for_frequency, 2)];
        outer_model_second_component_indices = outer_model_second_component_indices ...
            + ROI_model_second_component_indices(end);
        
        % Save the indices in the parameters structure
        params.ROI_model_first_component_indices = ROI_model_first_component_indices; % Indices in the combined model corresponding to the ROI model for the [Real; Imaginary] portion
        params.outer_model_first_component_indices = outer_model_first_component_indices; % Indices in the combined model corresponding to the outer model for the [Real; Imaginary] portion
        params.ROI_model_second_component_indices = ROI_model_second_component_indices;  % Indices in the combined model corresponding to the ROI model for the [-Imaginary; Real] portion
        params.outer_model_second_component_indices = outer_model_second_component_indices; % Indices in the combined model corresponding to the outer model for the [-Imaginary; Real] portion
        
        % Order the parameters structure by ASCII dictionary order
        params = orderfields(params);
        
        % Save the combined model, the ROI model predictor positions, the 
        % outer model predictor positions, and the parameters structure
        filename = ['STFT_window_number_' num2str(window_id) '_frequency_number_' num2str(freq_num) '.mat'];
        save(fullfile(params.models_save_path, filename), '-v7.3', 'combined_model', ...
            'ROI_predictor_x_z_distance_offset_matrix', 'outer_predictor_x_z_distance_offset_matrix', ...
            'ROI_model_space_matrix', 'outer_model_space_matrix', 'params');
    end

end