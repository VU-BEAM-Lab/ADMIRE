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


% Description of generate_modeled_signal_for_predictor.m:
% This function generates the modeled signal for a single model predictor


function modeled_signal_for_predictor = generate_modeled_signal_for_predictor(freq_num, predictor_x_z_distance_offset, z_center_stft_window, params)

    %% Parameter Calculation %%
    % Define the lateral position of the predictor
    predictor_x = predictor_x_z_distance_offset(1);

    % Define the depth position of the predictor
    predictor_z = predictor_x_z_distance_offset(2);

    % Define the distance offset for the predictor
    predictor_distance_offset = predictor_x_z_distance_offset(3);
    predictor_distance_offset = predictor_distance_offset + params.distance_offset_shift;
    
    
    %% Element Sensitivity Calculation %%
    % Calculate the element sensitivity across the aperture for the predictor
    theta = atan2(params.elem_pos_x - predictor_x, predictor_z);
    wavelength = (2 .* pi) ./ (params.k_calibrated(freq_num));
    argum = (pi .* params.probe_pitch .* sin(theta)) ./ wavelength;
    A_es = ((sin(argum) + eps) ./ (argum + eps)) .* cos(theta);
 
    
    %% STFT Window Amplitude Modulation Calculation %%
    % Calculate the perceived depth across the aperture for the predictor
    tau_n0 = (predictor_distance_offset + (2 .* z_center_stft_window) - (predictor_z - params.cal_shift)) ./ params.c;
    d0_x = sqrt(((params.elem_pos_x - predictor_x) .^ 2) + ((predictor_z - params.cal_shift) .^ 2)) + (params.c .* tau_n0);
    if params.gamma_im == 0
        perceived_depth = ((d0_x .^ 2) - ((params.elem_pos_x) .^ 2)) ./ (2 .* d0_x);
        second_component = (sqrt(((params.elem_pos_x) .^ 2) + (perceived_depth .^ 2)) - perceived_depth) ./ params.c;
    else
        perceived_depth = (0.5 .* (d0_x .* (1 - params.gamma_im) ./ (2 .* params.gamma_im))) ...
            - sign(params.gamma_im) .* 0.5 .* sqrt(((d0_x .* (1 - params.gamma_im) ./ (2 .* params.gamma_im)) .^ 2) ...
            - ((((params.elem_pos_x) .^ 2) - (d0_x .^ 2)) ./ params.gamma_im));
        second_component = (sqrt(((params.elem_pos_x) .^ 2) + ((perceived_depth .* (1 + params.gamma_im)) .^ 2)) ...
            - abs(perceived_depth .* (1 + params.gamma_im))) ./ params.c;
    end
   
    % Calculate the wavefront delays across the aperture for the predictor
    % The second component variable accounts for all of the terms that are
    % subtracted from tau_Diff
    tau_Diff = d0_x ./ params.c;
    tau = tau_Diff - second_component;
    
    % Convert the tau values to distance values
    z_distance = tau .* params.c;
    
    % Determine the distance that a sound wave travels to the shallowest
    % depth in the STFT window and back to a depth of 0 m
    min_travel_distance_STFT_window = params.depths_in_window(1) .* 2;
    
    % Determine the distance that a sound wave travels to the deepest depth
    % in the STFT window and back to a depth of 0 m
    max_travel_distance_STFT_window = params.depths_in_window(end) .* 2;
    
    % Calculate how much of the wavefront for the predictor lies within the 
    % sound wave travel distance range for the STFT window (there is an axial 
    % Gaussian pulse for each element position, and z_distance contains a 
    % vector of values that correspond to the distance that the center of
    % each Gaussian pulse has traveled)
    half_pulse_length_distance = params.half_pulse_width_samples ./ (params.fs) .* (params.c ./ 2);
    pulse_min_travel_distance_values = z_distance - (half_pulse_length_distance .* params.win_tune);
    pulse_min_travel_distance_values(pulse_min_travel_distance_values < min_travel_distance_STFT_window) = min_travel_distance_STFT_window;
    pulse_max_travel_distance_values = z_distance + (half_pulse_length_distance .* params.win_tune);
    pulse_max_travel_distance_values(pulse_max_travel_distance_values > max_travel_distance_STFT_window) = max_travel_distance_STFT_window;
    sff = ((params.BW .* params.f0) .^ 2) ./ (8 .* log(2));
    stt = ((params.c) .^ 2) ./ (4 .* (pi .^ 2) .* sff);
    st = sqrt(stt);
    
    % Perform analytic Gaussian window integration and account for the case
    % when the pulse travel distance range does not lie within the sound
    % wave travel distance range for the STFT window
    gaussian_analytic = 0.5 .* sqrt(pi) .* st .* (erf((z_distance - pulse_min_travel_distance_values) ./ st) ...
        - erf((z_distance - pulse_max_travel_distance_values) ./ st));
    gaussian_analytic(pulse_min_travel_distance_values > max_travel_distance_STFT_window ...
        | pulse_max_travel_distance_values < min_travel_distance_STFT_window) = 0;
    A_FT = sqrt(gaussian_analytic ./ max(gaussian_analytic));
    
    
    %% Phase Modulation Calculation %%
    % Calculate the phase modulation terms for the predictor across the
    % aperture
    phase_modulation = exp(1i .* z_distance .* params.k_calibrated(freq_num));
    
    
    %% Modeled Signal Calculation %%
    % Calculate the modeled signal for the predictor
    modeled_signal_for_predictor = A_FT .* A_es .* phase_modulation;
    
end