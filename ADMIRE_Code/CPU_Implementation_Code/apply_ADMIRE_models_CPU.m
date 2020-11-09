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


% Description of apply_ADMIRE_models_CPU.m:
% This function applies the ADMIRE models to channel data on a CPU


function apply_ADMIRE_models_CPU(apply_params)
        
    %% Load In Channel Data And Parameters %%
    % Load in the channel data
    channel_data = load(fullfile(apply_params.data_load_path, apply_params.filename));
    name = fieldnames(channel_data);
    channel_data = channel_data.(name{1});    
    
    % Convert the channel data to double precision
    channel_data = double(channel_data);
    
    % Load in the parameters structures that was used to generate the
    % ADMIRE models
    filename = ['STFT_window_number_1_frequency_number_1.mat'];
    load(fullfile(apply_params.models_load_path, filename));
    
    
    %% Reshape Channel Data %%
    % Reshape the channel data if the data type is 'Verasonics RF Buffer'
    % or don't reshaped if the data type is 'Reshaped'
    if strcmp(params.data_type, 'Verasonics RF Buffer')
        num_receive_samples = params.num_beams .* (params.num_depths + params.t0 - 1);
        channel_data = channel_data(1:num_receive_samples, :, :);
        channel_data = reshape(channel_data, [(params.num_depths + params.t0 - 1), ...
            params.num_beams, params.total_elements_on_probe, size(channel_data, 3)]);
        channel_data = permute(channel_data, [1 3 2 4]);
        channel_data = channel_data(params.t0:end, :, :, :);
        
        % Obtain the elements for each beam (walked aperture)
        channel_data_reshaped = zeros(size(channel_data, 1), params.num_elements, ...
            params.num_beams, size(channel_data, 4));
        for beam_ind = 1:params.num_beams
            channel_data_reshaped(:, :, beam_ind, :) = channel_data(:, beam_ind:(beam_ind + params.num_elements - 1), beam_ind, :);
        end
    elseif strcmp(params.data_type, 'Reshaped')
        channel_data_reshaped = channel_data(params.t0:end, :, :, :);
    end
    
    
    %% Calculate Time Delays %%
    % Calculate the time delays if the probe type is 'Linear'
    if strcmp(params.probe_type, 'Linear')
        % Calculate the element positions for one beam
        elem_pos_x = [0:params.num_elements - 1] .* params.probe_pitch;   
        elem_pos_x = elem_pos_x - mean(elem_pos_x);  % Lateral positions of the elements (m) 
        
        % Calculate the delays that are applied to the channel data
        elem_pos_x_matrix = repmat(elem_pos_x, [params.num_depths, 1]);
        depths_matrix = repmat(params.depths', [1, params.num_elements]);
        delays = ((1 ./ params.c) .* sqrt((elem_pos_x_matrix .^ 2) + (depths_matrix .^ 2))) + (depths_matrix ./ params.c);
        delays = repmat(delays, [1, 1, params.num_beams]);  % Delays for the channel data (s)
    elseif strcmp(params.probe_type, 'Curvilinear')
        % Calculate the time delays if the probe type is 'Curvilinear'
        % Calculate the element positions for one beam
        e = [0:params.total_elements_on_probe - 1] - mean([0:params.total_elements_on_probe - 1]);
        elem_pos_x = params.probe_radius .* sin(params.dtheta .* e);  % Lateral positions of all of the elements on the transducer (m)
        elem_pos_z = params.probe_radius .* cos(params.dtheta .* e);  % Axial positions of all of the elements on the transducer (m) 
        
        % Calculate the delays that are applied to the channel data
        center_depths = params.probe_radius + params.depths;
        beam = [0:params.num_beams - 1] - mean([0:params.num_beams - 1]);
        delays = zeros(params.num_depths, params.num_elements, params.num_beams);
        for beam_ind = 1:params.num_beams
            bx = center_depths .* sin(params.dtheta .* beam(beam_ind));
            bz = center_depths .* cos(params.dtheta .* beam(beam_ind));
            bx = repmat(bx', [1, params.num_elements]);
            bz = repmat(bz', [1, params.num_elements]);
            elem_pos_x_beam = elem_pos_x(beam_ind:(beam_ind + params.num_elements - 1));
            elem_pos_z_beam = elem_pos_z(beam_ind:(beam_ind + params.num_elements - 1));
            elem_pos_x_beam = repmat(elem_pos_x_beam, [params.num_depths, 1]);
            elem_pos_z_beam = repmat(elem_pos_z_beam, [params.num_depths, 1]);
            delays(:, :, beam_ind) = ((1 ./ params.c) .* sqrt(((elem_pos_x_beam - bx) .^ 2) ...
                + ((elem_pos_z_beam - bz) .^ 2))) + ((sqrt((bx .^ 2) + (bz .^ 2)) ...
                - params.probe_radius) ./ params.c);
        end
    end
    
    
    %% Beamform Channel Data %%
    % Apply the time delays to each frame of the channel data
    beamformed_channel_data = zeros(size(channel_data_reshaped));
    num_frames = size(channel_data_reshaped, 4);
    times_array = ([0:(params.num_depths - 1)] ./ params.fs)';
    for frame_ind = 1:num_frames
        for beam_ind = 1:params.num_beams
            for elem_ind = 1:params.num_elements
                beamformed_channel_data(:, elem_ind, beam_ind, frame_ind) = interp1(times_array, ...
                    channel_data_reshaped(:, elem_ind, beam_ind, frame_ind), delays(:, elem_ind, beam_ind), ...
                    'linear', 0);
            end
        end
    end

  
    %% Process Channel Data With ADMIRE %%
    % Determine the indices of the negative frequencies that correspond to
    % the positive selected frequencies
    num_selected_freqs = length(params.selected_freqs);
    num_corresponding_negative_freqs = num_selected_freqs - sum(params.selected_freqs == 0);
    negative_freq_include = params.selected_freqs ~= 0;
    negative_freq_inds = zeros(1, num_corresponding_negative_freqs);  % Indices of corresponding negative frequencies (1 is subtracted to account for zero-based indexing on the GPU) 
    if mod(params.zero_padded_stft_window_length, 2) == 1
        freqs = ([(-(params.zero_padded_stft_window_length - 1) ./ 2):((params.zero_padded_stft_window_length - 1) ./ 2)] ./ params.zero_padded_stft_window_length) .* params.fs;
        negative_half = freqs(1:((params.zero_padded_stft_window_length - 1) ./ 2));
        rearranged_freqs = [freqs((((params.zero_padded_stft_window_length - 1) ./ 2) + 1:end)) negative_half];
        count = 1;
        for ii = 1:num_selected_freqs
            if params.selected_freqs(ii) ~= 0
                ind = find(-params.selected_freqs(ii) == rearranged_freqs);
                negative_freq_inds(count) = ind;
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
                negative_freq_inds(count) = ind;
                count = count + 1;
            end
        end
    end
    
    % Calculate the windowing function coefficients
    windowing_function_values = window(params.stft_windowing_function, params.stft_window_length);
    windowing_function_values = repmat(windowing_function_values, [1, params.num_elements]);
    
    % Preallocate the matrix that will store the reconstructed channel data
    reconstructed_channel_data = beamformed_channel_data;
    
    % Apply ADMIRE to each frame of channel data
    for frame_ind = 1:num_frames
        for beam_ind = 1:params.num_beams
            for window_ind = 1:params.stft_num_windows
                % Print the processing progress
                fprintf('Processing STFT window #%d of beam #%d of frame #%d.\n', window_ind, beam_ind, frame_ind);
   
                % Obtain the data for the current STFT window
                start_depth_ind = params.stft_start_depth_inds(params.stft_window_inds(window_ind));
                stft_window_data = beamformed_channel_data(start_depth_ind:(start_depth_ind + params.stft_window_length - 1), :, beam_ind, frame_ind);
                stft_window_data = stft_window_data .* windowing_function_values;
                stft_window_data = fft(stft_window_data, params.zero_padded_stft_window_length);
                
                % Preallocate the matrix that will store the reconstructed
                % frequency data for the STFT window (the frequencies that
                % aren't fit are zeroed out)
                reconstructed_stft_window_data = zeros(size(stft_window_data));
                
                % Separate the real and imaginary components
                real_component = real(stft_window_data);
                imaginary_component = imag(stft_window_data);
                stft_window_data = [real_component, imaginary_component];
                
                % Loop through the selected frequencies within the STFT
                % window
                count = 1;
                for freq_ind = 1:num_selected_freqs
                    % Load in the model and parameters for the current
                    % frequency
                    filename = ['STFT_window_number_' num2str(window_ind) '_frequency_number_' num2str(freq_ind) '.mat'];
                    load(fullfile(apply_params.models_load_path, filename));
                    
                    % Apply aperture growth to the aperture data
                    aperture_growth_mask = repmat(params.aperture_growth_mask, [1, 2]);
                    frequency_data = stft_window_data(params.selected_freq_inds(freq_ind), :);
                    frequency_data = frequency_data(aperture_growth_mask == 1);
                    frequency_data = frequency_data';
                    
                    % Calculate the lambda value that is used for elastic-net regularization
                    lambda = params.lambda_scaling_factor .* sqrt(mean(frequency_data .^ 2)); 
                    
                    % Clear MEX to unload any MEX-functions from memory
                    clear mex;
                
                    % Fit the ADMIRE model to the data by using cyclic coordinate descent in order to perform
                    % least-squares regression with elastic-net regularization
                    beta = ccd_double_precision([size(combined_model, 1), size(combined_model, 2), ...
                        params.alpha, lambda, params.tolerance, params.max_iterations], combined_model, ...
                        frequency_data);                   

                    % Reconstruct the signal using the ROI predictors only
                    ROI_indices = [params.ROI_model_first_component_indices, params.ROI_model_second_component_indices];
                    ROI_model = combined_model(:, ROI_indices);
                    ROI_B = beta(ROI_indices);
                    reconstructed_frequency_data = zeros(1, length(aperture_growth_mask));
                    reconstructed_frequency_data(aperture_growth_mask == 1) = ROI_model * ROI_B;
                    
                    % Combine the real and imaginary components of the
                    % reconstructed frequency data
                    real_component_data = reconstructed_frequency_data(1:(length(aperture_growth_mask) ./ 2));
                    imaginary_component_data = reconstructed_frequency_data(((length(aperture_growth_mask) ./ 2) + 1):end);
                    reconstructed_frequency_data = real_component_data + (1i .* imaginary_component_data);
                    
                    % Replace the original frequency data with the
                    % reconstructed frequency data
                    reconstructed_stft_window_data(params.selected_freq_inds(freq_ind), :) = reconstructed_frequency_data;
                    
                    % Store the conjugate of the reconstructed frequency
                    % data for the corresponding negative frequency if
                    % there is one
                    if negative_freq_include(freq_ind) == 1
                        reconstructed_stft_window_data(negative_freq_inds(count), :) = conj(reconstructed_frequency_data);
                        count = count + 1;
                    end
                end
                
                % Take the inverse Fourier transform of each column of the
                % STFT window (complex values can potentially be returned
                % depending on the numerical error of the inverse Fourier 
                % transform, so only the real component is kept)
                reconstructed_stft_window_data = real(ifft(reconstructed_stft_window_data));
                
                % Remove the zeros corresponding to zero-padding
                reconstructed_stft_window_data = reconstructed_stft_window_data(1:params.stft_window_length, :);
                
                % Store the reconstructed channel data into the
                % reconstructed channel data matrix
                reconstructed_channel_data(start_depth_ind:(start_depth_ind + params.stft_window_length - 1), :, beam_ind, frame_ind) = reconstructed_stft_window_data;
            end
        end
    end
    
    
    %% Image Data Calculation %%
    % Calculate the image data from the channel data that was processed
    % with ADMIRE
    % Sum the channel data and calculate the envelope data
    summed_channel_data = squeeze(sum(reconstructed_channel_data, 2));
    envelope_data = abs(hilbert(summed_channel_data));
        
    % Normalize the envelope data for each frame
    for frame_ind = 1:num_frames
        frame_envelope_data = envelope_data(:, :, frame_ind);
        envelope_data(:, :, frame_ind) = frame_envelope_data ./ max(frame_envelope_data(:));
    end
        
    % Apply log compression to the envelope data 
    envelope_matrix = 20 .* log10(envelope_data);
    
    % Determine the axial and lateral positions (cm) for the normalized and
    % log-compressed envelope data
    axial_positions_cm = params.depths .* 100;
    lateral_positions = [0:(params.num_beams - 1)] .* params.probe_pitch;
    lateral_positions = lateral_positions - mean(lateral_positions);
    lateral_positions_cm = lateral_positions .* 100;
    
    % Perform scan conversion for the curvilinear probe case
    if strcmp(params.probe_type, 'Curvilinear')
        % Calculate the scan sector angle in degrees that each beam corresponds to
        sector = ([0:params.num_beams - 1] - mean([0:params.num_beams - 1])) .* params.dtheta .* (180 ./ pi);
        
        % Determine the minimum scan sector angle in degrees
        min_phi = sector(1);
        
        % Determine the angle span in degrees of the scan
        span_phi = sector(end) - sector(1);
    
        % Define the distance to the radial center of the probe in cm
        apex = -params.probe_radius .* 100;
        
        % Define the variable that determines whether to do nearest
        % neighbor interpolation or bilinear interpolation for scan
        % conversion (1 = nearest neighbor and 2 = bilinear interpolation)
        dsfactor = 2;
        
        % Define the array is used to determine the scan conversion grid
        vargin = [0 0 5E-5 0 0 5E-5];  % [ax_min, ax_max, ax_inc, lat_min, lat_max, lat_inc] (meters)
        
        % Obtain the dimensions of one frame of the scan-converted envelope
        % data in order to preallocate memory for storing all of the frames
        % of the scan-converted envelope data
        [test_data, ~, ~] = sconvert(envelope_matrix(:, :, 1), 'sector', min_phi, span_phi, apex, dsfactor, params.fs, params.c, vargin);
        envelope_matrix_scan_converted = zeros(size(test_data, 1), size(test_data, 2), num_frames);
        
        % Perform scan conversion for each frame using the sconvert and 
        % scmap functions
        for frame_ind = 1:num_frames
            [envelope_matrix_scan_converted(:, :, frame_ind), axial_positions_cm, lateral_positions_cm] = ...
                sconvert(envelope_matrix(:, :, frame_ind), 'sector', min_phi, span_phi, apex, dsfactor, params.fs, params.c, vargin);
        end
        
        % Store the scan-converted data into the matrix containing the
        % image data
        envelope_matrix = envelope_matrix_scan_converted;
    end
    
   
    %% Save Out The ADMIRE Results %%
    if params.channel_data_output_flag == 0
        save(fullfile(apply_params.processed_data_save_path, apply_params.processed_data_filename), ...
            '-v7.3', 'envelope_matrix', 'axial_positions_cm', 'lateral_positions_cm');
    elseif params.channel_data_output_flag == 1
        save(fullfile(apply_params.processed_data_save_path, apply_params.processed_data_filename), ...
            '-v7.3', 'envelope_matrix', 'reconstructed_channel_data', 'axial_positions_cm', ...
            'lateral_positions_cm'); 
    end
    
    
    %% Display The Images %%
    % Display each image frame if specified
    if apply_params.display_image_flag == 1
        for frame_ind = 1:num_frames
            figure;
            imagesc(lateral_positions_cm, axial_positions_cm, envelope_matrix(:, :, frame_ind));
            colormap gray;
            caxis(apply_params.display_caxis_limits);
            colorbar;
            axis image;
            xlabel('Lateral Position (cm)');
            ylabel('Depth (cm)');
            title(['Frame ' num2str(frame_ind)]);            
        end
    end
     
end
