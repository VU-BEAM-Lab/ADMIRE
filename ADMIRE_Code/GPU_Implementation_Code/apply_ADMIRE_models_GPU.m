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


% Description of apply_ADMIRE_models_GPU.m:
% This function applies the ADMIRE models to channel data on a GPU


function apply_ADMIRE_models_GPU(apply_params)
    
    %% Clear MEX %%
    % Clear MEX to unload any MEX-functions from memory
    clear mex;
    
    
    %% ADMIRE GPU Processing Of Data %%
    % Apply ADMIRE on the GPU depending on the probe and data type case
    if strcmp(apply_params.probe_type, 'Linear')
        if strcmp(apply_params.data_type, 'Reshaped')
            % Load in the ADMIRE variables
            filename = 'ADMIRE_Variables_Linear_Probe_Reshaped_Data_Type.mat';
            load(fullfile(apply_params.models_load_path, filename));
           
            % Load in the channel data
            channel_data = load(fullfile(apply_params.data_load_path, apply_params.filename));
            name = fieldnames(channel_data);
            channel_data = channel_data.(name{1});
            
            % Convert the channel data to single precision
            channel_data = single(channel_data);
            
            % Allocate the matrix that will hold the envelope data for each
            % frame
            num_depths = GPU_fixed_params(2);
            num_beams = GPU_fixed_params(4);
            envelope_matrix = single(zeros(num_depths, num_beams, size(channel_data, 4)));
            
            % This is for the case where the reconstructed channel data is
            % being outputted in addition to the envelope data for each
            % frame
            if GPU_fixed_params(end) == 1
                % Allocate the matrix that will hold the reconstructed 
                % channel data for each frame
                num_elements = GPU_fixed_params(3);
                reconstructed_channel_data = single(zeros(num_depths, num_elements, num_beams, size(channel_data, 4)));
                
                % Apply ADMIRE to each frame of data
                for frame_ind = 1:size(channel_data, 4)
                    data_frame = channel_data(:, :, :, frame_ind);
                    [envelope, ADMIRE_channel_data] = ADMIRE_GPU_linear_probe_reshaped_data_type(GPU_fixed_params, ...
                        delays, stft_window, selected_freq_inds, negative_freq_inds, ...
                        negative_freq_include, y_include_mask, num_observations, ...
                        observation_thread_stride, num_predictors, X_matrix_thread_stride, ...
                        X_matrix, B_thread_stride, GPU_adjustable_params, data_frame);
                    
                    % Store the envelope data for the current frame
                    envelope_matrix(:, :, frame_ind) = envelope;
                    
                    % Store the reconstructed channel data for the current frame
                    reconstructed_channel_data(:, :, :, frame_ind) = reshape(ADMIRE_channel_data, [num_depths, num_elements, num_beams]);
                end
            elseif GPU_fixed_params(end) == 0
                % This is for the case where only the envelope data is
                % being outputted for each frame
                % Apply ADMIRE to each frame of data
                for frame_ind = 1:size(channel_data, 4)
                    data_frame = channel_data(:, :, :, frame_ind);
                    [envelope] = ADMIRE_GPU_linear_probe_reshaped_data_type(GPU_fixed_params, ...
                        delays, stft_window, selected_freq_inds, negative_freq_inds, ...
                        negative_freq_include, y_include_mask, num_observations, ...
                        observation_thread_stride, num_predictors, X_matrix_thread_stride, ...
                        X_matrix, B_thread_stride, GPU_adjustable_params, data_frame);
                    
                    % Store the envelope data for the current frame
                    envelope_matrix(:, :, frame_ind) = envelope;   
                end
            end
        elseif strcmp(apply_params.data_type, 'Verasonics RF Buffer')
            % Load in the ADMIRE variables
            filename = 'ADMIRE_Variables_Linear_Probe_Verasonics_RF_Buffer_Data_Type.mat';
            load(fullfile(apply_params.models_load_path, filename));
            
            % Load in the channel data
            channel_data = load(fullfile(apply_params.data_load_path, apply_params.filename));
            name = fieldnames(channel_data);
            channel_data = channel_data.(name{1});           
            
            % Allocate the matrix that will hold the envelope data for each
            % frame
            num_depths = GPU_fixed_params(3);
            num_beams = GPU_fixed_params(6);
            envelope_matrix = single(zeros(num_depths, num_beams, size(channel_data, 4)));  
            
            % This is for the case where the reconstructed channel data is
            % being outputted in addition to the envelope data for each
            % frame
            if GPU_fixed_params(end) == 1
                % Allocate the matrix that will hold the reconstructed 
                % channel data for each frame
                num_elements = GPU_fixed_params(5);
                reconstructed_channel_data = single(zeros(num_depths, num_elements, num_beams, size(channel_data, 4)));
                
                % Apply ADMIRE to each frame of data
                for frame_ind = 1:size(channel_data, 4)
                    data_frame = channel_data(:, :, :, frame_ind);
                    [envelope, ADMIRE_channel_data] = ADMIRE_GPU_linear_probe_Verasonics_RF_buffer_data_type(GPU_fixed_params, ...
                        delays, stft_window, selected_freq_inds, negative_freq_inds, ...
                        negative_freq_include, y_include_mask, num_observations, ...
                        observation_thread_stride, num_predictors, X_matrix_thread_stride, ...
                        X_matrix, B_thread_stride, GPU_adjustable_params, data_frame);
                    
                    % Store the envelope data for the current frame
                    envelope_matrix(:, :, frame_ind) = envelope;
                    
                    % Store the reconstructed channel data for the current frame
                    reconstructed_channel_data(:, :, :, frame_ind) = reshape(ADMIRE_channel_data, [num_depths, num_elements, num_beams]);
                end            
            elseif GPU_fixed_params(end) == 0
                % This is for the case where only the envelope data is
                % being outputted for each frame
                % Apply ADMIRE to each frame of data
                for frame_ind = 1:size(channel_data, 4)
                    data_frame = channel_data(:, :, :, frame_ind);
                    [envelope] = ADMIRE_GPU_linear_probe_Verasonics_RF_buffer_data_type(GPU_fixed_params, ...
                        delays, stft_window, selected_freq_inds, negative_freq_inds, ...
                        negative_freq_include, y_include_mask, num_observations, ...
                        observation_thread_stride, num_predictors, X_matrix_thread_stride, ...
                        X_matrix, B_thread_stride, GPU_adjustable_params, data_frame);
                    
                    % Store the envelope data for the current frame
                    envelope_matrix(:, :, frame_ind) = envelope;   
                end
            end
        end
    elseif strcmp(apply_params.probe_type, 'Curvilinear')
        if strcmp(apply_params.data_type, 'Reshaped')
            % Load in the ADMIRE variables
            filename = 'ADMIRE_Variables_Curvilinear_Probe_Reshaped_Data_Type.mat';
            load(fullfile(apply_params.models_load_path, filename));
            
            % Load in the channel data
            channel_data = load(fullfile(apply_params.data_load_path, apply_params.filename));
            name = fieldnames(channel_data);
            channel_data = channel_data.(name{1});  
            
            % Convert the channel data to single precision
            channel_data = single(channel_data);
            
            % Allocate the matrix that will hold the envelope data for each
            % frame
            num_depths = GPU_fixed_params(2);
            num_beams = GPU_fixed_params(4);
            scan_converted_num_axial_positions = GPU_fixed_params(18);
            scan_converted_num_lateral_positions = GPU_fixed_params(19);
            envelope_matrix = single(zeros(scan_converted_num_axial_positions, scan_converted_num_lateral_positions, ...
                size(channel_data, 4))); 
            
            % This is for the case where the reconstructed channel data is
            % being outputted in addition to the scan-converted envelope data 
            % for each frame
            if GPU_fixed_params(end) == 1
                % Allocate the matrix that will hold the reconstructed 
                % channel data for each frame
                num_elements = GPU_fixed_params(3);
                reconstructed_channel_data = single(zeros(num_depths, num_elements, num_beams, size(channel_data, 4)));
                
                % Apply ADMIRE to each frame of data
                for frame_ind = 1:size(channel_data, 4)
                    data_frame = channel_data(:, :, :, frame_ind);
                    [envelope, ADMIRE_channel_data] = ADMIRE_GPU_curvilinear_probe_reshaped_data_type(GPU_fixed_params, ...
                        delays, stft_window, selected_freq_inds, negative_freq_inds, ...
                        negative_freq_include, y_include_mask, num_observations, ...
                        observation_thread_stride, num_predictors, X_matrix_thread_stride, ...
                        X_matrix, B_thread_stride, dr, dth, i00, i01, i10, ...
                        i11, idx, GPU_adjustable_params, data_frame);
               
                    % Store the envelope data for the current frame
                    envelope_matrix(:, :, frame_ind) = envelope;
                    
                    % Store the reconstructed channel data for the current frame
                    reconstructed_channel_data(:, :, :, frame_ind) = reshape(ADMIRE_channel_data, [num_depths, num_elements, num_beams]);
                end  
            elseif GPU_fixed_params(end) == 0
                % This is for the case where only the scan-converted envelope 
                % data is being outputted for each frame
                % Apply ADMIRE to each frame of data
                for frame_ind = 1:size(channel_data, 4)
                    data_frame = channel_data(:, :, :, frame_ind);
                    [envelope] = ADMIRE_GPU_curvilinear_probe_reshaped_data_type(GPU_fixed_params, ...
                        delays, stft_window, selected_freq_inds, negative_freq_inds, ...
                        negative_freq_include, y_include_mask, num_observations, ...
                        observation_thread_stride, num_predictors, X_matrix_thread_stride, ...
                        X_matrix, B_thread_stride, dr, dth, i00, i01, i10, ...
                        i11, idx, GPU_adjustable_params, data_frame);
                    
                    % Store the envelope data for the current frame
                    envelope_matrix(:, :, frame_ind) = envelope;   
                end
            end
        elseif strcmp(apply_params.data_type, 'Verasonics RF Buffer')
            % Load in the ADMIRE variables
            filename = 'ADMIRE_Variables_Curvilinear_Probe_Verasonics_RF_Buffer_Data_Type.mat';
            load(fullfile(apply_params.models_load_path, filename));
            
            % Load in the channel data
            channel_data = load(fullfile(apply_params.data_load_path, apply_params.filename));
            name = fieldnames(channel_data);
            channel_data = channel_data.(name{1});            
            
            % Allocate the matrix that will hold the envelope data for each
            % frame
            num_depths = GPU_fixed_params(3);
            num_beams = GPU_fixed_params(6);
            scan_converted_num_axial_positions = GPU_fixed_params(20);
            scan_converted_num_lateral_positions = GPU_fixed_params(21);
            envelope_matrix = single(zeros(scan_converted_num_axial_positions, scan_converted_num_lateral_positions, ...
                size(channel_data, 4)));  
            
            % This is for the case where the reconstructed channel data is
            % being outputted in addition to the scan-converted envelope data 
            % for each frame
            if GPU_fixed_params(end) == 1
                % Allocate the matrix that will hold the reconstructed 
                % channel data for each frame
                num_elements = GPU_fixed_params(5);
                reconstructed_channel_data = single(zeros(num_depths, num_elements, num_beams, size(channel_data, 4)));
                
                % Apply ADMIRE to each frame of data
                for frame_ind = 1:size(channel_data, 4)
                    data_frame = channel_data(:, :, :, frame_ind);
                    [envelope, ADMIRE_channel_data] = ADMIRE_GPU_curvilinear_probe_Verasonics_RF_buffer_data_type(GPU_fixed_params, ...
                        delays, stft_window, selected_freq_inds, negative_freq_inds, ...
                        negative_freq_include, y_include_mask, num_observations, ...
                        observation_thread_stride, num_predictors, X_matrix_thread_stride, ...
                        X_matrix, B_thread_stride, dr, dth, i00, i01, i10, ...
                        i11, idx, GPU_adjustable_params, data_frame);
                    
                    % Store the envelope data for the current frame
                    envelope_matrix(:, :, frame_ind) = envelope;
                    
                    % Store the reconstructed channel data for the current frame
                    reconstructed_channel_data(:, :, :, frame_ind) = reshape(ADMIRE_channel_data, [num_depths, num_elements, num_beams]);
                end
            elseif GPU_fixed_params(end) == 0
                % This is for the case where only the scan-converted envelope 
                % data is being outputted for each frame
                % Apply ADMIRE to each frame of data
                for frame_ind = 1:size(channel_data, 4)
                    data_frame = channel_data(:, :, :, frame_ind);
                    [envelope] = ADMIRE_GPU_curvilinear_probe_Verasonics_RF_buffer_data_type(GPU_fixed_params, ...
                        delays, stft_window, selected_freq_inds, negative_freq_inds, ...
                        negative_freq_include, y_include_mask, num_observations, ...
                        observation_thread_stride, num_predictors, X_matrix_thread_stride, ...
                        X_matrix, B_thread_stride, dr, dth, i00, i01, i10, ...
                        i11, idx, GPU_adjustable_params, data_frame);
                    
                    % Store the envelope data for the current frame
                    envelope_matrix(:, :, frame_ind) = envelope;   
                end
            end
        end
    end
    
  
    %% Display Of Images %%
    % Display the images if specified
    if apply_params.display_image_flag == 1
        if strcmp(apply_params.probe_type, 'Linear')
            axial_positions_cm = depths .* 100;
            lateral_positions_cm = beam_pos_x .* 100;
            for frame_ind = 1:size(envelope_matrix, 3)
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
        elseif strcmp(apply_params.probe_type, 'Curvilinear')
            axial_positions_cm = ax;
            lateral_positions_cm = la;
            for frame_ind = 1:size(envelope_matrix, 3)
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
    
    
    %% Save Out Processed Data %%
    if GPU_fixed_params(end) == 1
        save(fullfile(apply_params.processed_data_save_path, apply_params.processed_data_filename), ...
            '-v7.3', 'envelope_matrix', 'reconstructed_channel_data', 'axial_positions_cm', ...
            'lateral_positions_cm');
    elseif GPU_fixed_params(end) == 0
        save(fullfile(apply_params.processed_data_save_path, apply_params.processed_data_filename), ...
            '-v7.3', 'envelope_matrix', 'axial_positions_cm', 'lateral_positions_cm');
    end
    
end