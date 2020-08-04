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


% Description of parameter_values_check.m:
% This function checks to make sure that invalid parameter values are not
% entered


function params = parameter_values_check(params)

    % Check the processor_type parameter
    if ~strcmp(params.processor_type, 'CPU') && ~strcmp(params.processor_type, 'GPU')
        error('params.processor_type must be set to either ''CPU'' or ''GPU''.');
    end
    
    % Check the data_type parameter
    if ~strcmp(params.data_type, 'Reshaped') && ~strcmp(params.data_type, 'Verasonics RF Buffer')
        error('params.data_type must be set to either ''Reshaped'' or ''Verasonics RF Buffer''.');
    end
    
    % Check the t0 parameter
    if params.t0 <= 0
        error('params.t0 must be greater than 0.');
    end
    
    % Check the c parameter
    if params.c < 0
        error('params.c must be greater than or equal to 0.');
    end
    
    % Check the num_buffer_rows parameter
    if strcmp(params.data_type, 'Verasonics RF Buffer')
        if params.num_buffer_rows <= 0
            error('params.num_buffer_rows must be greater than 0.');
        end
    end
    
    % Check the num_depths parameter
    if params.num_depths <= 0
        error('params.num_depths must be greater than 0.');
    end
    
    % Check the num_elements parameter
    if params.num_elements <= 0
        error('params.num_elements must be greater than 0.');
    end
    
    % Check the num_beams parameter
    if params.num_beams <= 0
        error('params.num_beams must be greater than 0.');
    end
    
    % Check the total_elements_on_probe parameter
    if params.total_elements_on_probe <= 0
        error('params.total_elements_on_probe must be greater than 0.');
    end
    
    % Check the probe_type parameter
    if ~strcmp(params.probe_type, 'Linear') && ~strcmp(params.probe_type, 'Curvilinear')
        error('params.probe_type must be set to either ''Linear'' or ''Curvilinear''.');
    end
    
    % Check the probe_radius parameter
    if strcmp(params.probe_type, 'Curvilinear')
        if params.probe_radius < 0
            error('params.probe_radius must be greater than or equal to 0.');
        end
    end
    
    % Check the dtheta parameter
    if strcmp(params.probe_type, 'Curvilinear')
        if params.dtheta <= 0
            error('params.dtheta must be greater than 0.');
        end
    end
    
    % Check the probe_pitch parameter
    if params.probe_pitch <= 0
        error('params.probe_pitch must be greater than 0.');
    end
    
    % Check the start_depth_ADMIRE parameter
    if params.start_depth_ADMIRE < min(params.depths)
        fprintf('params.start_depth_ADMIRE is less than min(params.depths), which is %f m.\n', min(params.depths));
        fprintf('Setting params.start_depth_ADMIRE to %f m.\n', min(params.depths));
        params.start_depth_ADMIRE = min(params.depths);
    end
    
    % Check the end_depth_ADMIRE parameter
    if params.end_depth_ADMIRE > max(params.depths)
        fprintf('params.end_depth_ADMIRE is greater than max(params.depths), which is %f m.\n', max(params.depths));
        fprintf('Setting params.end_depth_ADMIRE to %f m.\n', max(params.depths));
        params.end_depth_ADMIRE = max(params.depths);        
    end
    
    % Check the max_iterations parameter
    if params.max_iterations < 0
        error('params.max_iterations must be greater than or equal to 0.');
    end
    
    % Check the tolerance parameter
    if params.tolerance < 0
        error('params.tolerance must be greater than or equal to 0.');
    end
    
    % Check the ICA_ADMIRE_flag parameter
    if params.ICA_ADMIRE_flag ~= 0 && params.ICA_ADMIRE_flag ~= 1
        error('params.ICA_ADMIRE_flag must be set to either 0 or 1.');
    end
    
    % Check the channel_data_output_flag parameter
    if params.channel_data_output_flag ~= 0 && params.channel_data_output_flag ~= 1
        error('params.channel_data_output_flag must be set to either 0 or 1.');
    end
    
    % Check the aperture_growth_flag parameter
    if params.aperture_growth_flag ~= 0 && params.aperture_growth_flag ~= 1
        error('params.aperture_growth_flag must be set to either 0 or 1.');
    end
    
    % Check the min_num_elements parameter
    if params.aperture_growth_flag == 1
        if params.min_num_elements <= 0
            error('params.min_num_elements must be greater than 0.');
        end
    end
    
    % Check the F_number parameter
    if params.aperture_growth_flag == 1
        if params.F_number <= 0
            error('params.F_number must be greater than 0.');
        end
    end
    
    % Check the stft_windowing_function parameter
    if ~strcmp(func2str(params.stft_windowing_function), 'rectwin')
        error('params.stft_windowing_function must be set to @rectwin.');
    end
        
    % Check the stft_window_overlap parameter
    if params.stft_window_overlap ~= 0
        error('params.stft_window_overlap must be 0 in this version of the code.');
    end
    
    % Check the gamma_im parameter
    if params.gamma_im ~= 0
        error('params.gamma_im must be 0 in this version of the code.');
    end
    
    % Check to make sure that the ICA_ADMIRE_flag is valid when
    % processor_type = 'GPU'
    if strcmp(params.processor_type, 'GPU')
        if params.ICA_ADMIRE_flag ~= 1
            error('params.ICA_ADMIRE_flag must be set to 1 when params.processor_type = ''GPU''.');
        end
    end

end