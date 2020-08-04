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


% Description of generate_model_for_frequency.m:
% This function generates the model for the defined model space for one
% frequency


function [model_for_frequency, predictor_x_z_distance_offset_matrix] = generate_model_for_frequency(freq_num, model_space_matrix, z_center_stft_window, params)

    %% Parameter Calculation %%
    % Determine the possible lateral locations for the model space
    model_space_x_positions = [model_space_matrix(1, 1):model_space_matrix(1, 2):model_space_matrix(1, 3)];
    
    % Determine the possible depth positions for the model space
    model_space_z_positions = [model_space_matrix(2, 1):model_space_matrix(2, 2):model_space_matrix(2, 3)];
    
    % Determine the possible distance offsets for the model space
    model_space_distance_offsets = [model_space_matrix(3, 1):model_space_matrix(3, 2):model_space_matrix(3, 3)];
    
    % Create a meshgrid for the model space
    [X, Z, Distance_Offsets] = meshgrid(model_space_x_positions, model_space_z_positions, model_space_distance_offsets);
    
    % Define the lateral positions for the predictors
    x_predictors = X(:);
    
    % Define the depth positions for the predictors
    z_predictors = Z(:);
    
    % Define the distance offsets for the predictors
    distance_offset_predictors = Distance_Offsets(:);
    
    
    %% Generate The Modeled Signal For Each Model Predictor %%
    % Determine the number of model predictors
    num_model_predictors = length(x_predictors);
    
    % Pre-allocate the model matrix and the matrix that stores the
    % predictor lateral positions, depth positions, and distance offsets
    model_for_frequency = zeros(params.num_elements_stft_window, num_model_predictors);
    predictor_x_z_distance_offset_matrix = zeros(num_model_predictors, 3);
   
    % Loop through the model predictors and generate the modeled signal for
    % each predictor (only include the modeled signals that are not all 0
    % or NaN)
    count = 1;
    for predictor_ind = 1:num_model_predictors
        predictor_x_z_distance_offset = [x_predictors(predictor_ind), z_predictors(predictor_ind), distance_offset_predictors(predictor_ind)];
        modeled_signal_for_predictor = generate_modeled_signal_for_predictor(freq_num, predictor_x_z_distance_offset, z_center_stft_window, params);
        if ~all(modeled_signal_for_predictor == 0) && ~any(isnan(modeled_signal_for_predictor))
            predictor_x_z_distance_offset_matrix(count, :) = predictor_x_z_distance_offset;
            model_for_frequency(:, count) = modeled_signal_for_predictor;
            count = count + 1;
        end
    end
    predictor_x_z_distance_offset_matrix = predictor_x_z_distance_offset_matrix(1:(count - 1), :);
    model_for_frequency = model_for_frequency(:, 1:(count - 1));
    
end