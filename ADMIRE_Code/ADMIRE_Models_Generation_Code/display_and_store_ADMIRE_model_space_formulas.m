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


% Description of display_and_store_ADMIRE_model_space_formulas.m:
% This function displays the formulas that are used to calculate the ROI 
% and outer model spaces for ADMIRE, and it stores the formulas into the
% parameters structure


function params = display_and_store_ADMIRE_model_space_formulas(params)

    %% Define And Display The ROI Model Space Formulas For ADMIRE %%
    % Obtain the scaling factors that are used to calculate the
    % possible lateral positions of the ROI model space
    ROI_scale_1_x = params.ROI_model_x_position_scaling_factors(1);
    ROI_scale_2_x = params.ROI_model_x_position_scaling_factors(2);
    ROI_scale_3_x = params.ROI_model_x_position_scaling_factors(3);

    % Obtain the scaling factors that are used to calculate the
    % possible depth positions of the ROI model space
    ROI_scale_1_z = params.ROI_model_z_position_scaling_factors(1);
    ROI_scale_2_z = params.ROI_model_z_position_scaling_factors(2);
    ROI_scale_3_z = params.ROI_model_z_position_scaling_factors(3);
    
    % Obtain the scaling factor and the constants that are used to
    % calculate the possible distance offsets for the ROI model space
    ROI_scale_1_distance_offset = params.ROI_model_distance_offset_scaling_factor;
    ROI_constant_1_distance_offset = params.ROI_model_distance_offset_constants(1);
    ROI_constant_2_distance_offset = params.ROI_model_distance_offset_constants(2);
    
    % Define the ROI model space formulas for ADMIRE
    ROI_model_lateral_positions_formula = sprintf('ROI model lateral positions [min, spacing, max]: [%0.4f * res_lat, %0.4f * res_lat, %0.4f * res_lat]', ...
        ROI_scale_1_x, ROI_scale_2_x, ROI_scale_3_x);
    ROI_model_depth_positions_formula = sprintf('ROI model depth positions [min, spacing, max]: [z_center_stft_window + (%0.4f * res_axl), %0.4f * res_axl, z_center_stft_window + (%0.4f * res_axl)]', ...
        ROI_scale_1_z, ROI_scale_2_z, ROI_scale_3_z);
    ROI_model_distance_offsets_formula = sprintf('ROI model distance offsets [min, spacing, max]: [%0.4f, (%0.4f * 2 .* pi) / k_calibrated, %0.4f]', ...
        ROI_constant_1_distance_offset, ROI_scale_1_distance_offset, ROI_constant_2_distance_offset);

    % Display the ROI model space formulas for ADMIRE
    disp(ROI_model_lateral_positions_formula);
    disp(ROI_model_depth_positions_formula);
    disp(ROI_model_distance_offsets_formula);
    
    
    %% Define And Display The Outer Model Space Formulas For ADMIRE %%
    % Obtain the scaling factors that are used to calculate the
    % possible lateral positions of the outer model space
    outer_scale_1_x = params.outer_model_x_position_scaling_factors(1);
    outer_scale_2_x = params.outer_model_x_position_scaling_factors(2);
    outer_scale_3_x = params.outer_model_x_position_scaling_factors(3);
    
    % Obtain the scaling factors and constant that are used to
    % calculate the possible depth positions of the outer model space
    outer_scale_1_z = params.outer_model_z_position_scaling_factors(1);
    outer_scale_2_z = params.outer_model_z_position_scaling_factors(2);
    outer_scale_3_z = params.outer_model_z_position_scaling_factors(3);
    outer_scale_4_z = params.outer_model_z_position_scaling_factors(4);
    outer_scale_5_z = params.outer_model_z_position_scaling_factors(5);
    outer_scale_6_z = params.outer_model_z_position_scaling_factors(6);
    outer_constant_1_z = params.outer_model_z_position_constants(1);
    
    % Obtain the scaling factor and constants that are used to
    % calculate the possible distance offsets for the outer model space
    outer_scale_1_distance_offset = params.outer_model_distance_offset_scaling_factor(1);
    outer_constant_1_distance_offset = params.outer_model_distance_offset_constants(1);
    outer_constant_2_distance_offset = params.outer_model_distance_offset_constants(2);
    
    % Define the outer model space formulas for ADMIRE
    outer_model_lateral_positions_formula = sprintf('Outer model lateral positions [min, spacing, max]: [%0.4f * lateral_limit, %0.4f * res_lat, %0.4f * lateral_limit]', ...
        outer_scale_1_x, outer_scale_2_x, outer_scale_3_x);
    outer_model_depth_positions_formula = sprintf('Outer model depth positions [min, spacing, max]: [(%0.4f * z_center_stft_window) + (%0.4f * res_axl) + %0.4f, %0.4f * res_axl, (%0.4f * z_center_stft_window) + (%0.4f * res_axl) + (%0.4f * z_center_stft_window)]', ...
        outer_scale_1_z, outer_scale_2_z, outer_constant_1_z, outer_scale_3_z, outer_scale_4_z, outer_scale_5_z, outer_scale_6_z);
    outer_model_distance_offsets_formula = sprintf('Outer model distance offsets [min, spacing, max]: [%0.4f, (%0.4f * 2 .* pi) / k_calibrated, %0.4f]', ...
        outer_constant_1_distance_offset, outer_scale_1_distance_offset, outer_constant_2_distance_offset);
    
    % Display the outer model space formulas for ADMIRE
    disp(outer_model_lateral_positions_formula);
    disp(outer_model_depth_positions_formula);
    disp(outer_model_distance_offsets_formula);
    
    
    %% Store The ROI And Outer Model Space Formulas For ADMIRE %%
    % Store the ADMIRE model space formulas into the parameters structure
    formulas_cell = cell(1, 6);
    formulas_cell{1} = ROI_model_lateral_positions_formula;
    formulas_cell{2} = ROI_model_depth_positions_formula;
    formulas_cell{3} = ROI_model_distance_offsets_formula;
    formulas_cell{4} = outer_model_lateral_positions_formula;
    formulas_cell{5} = outer_model_depth_positions_formula;
    formulas_cell{6} = outer_model_distance_offsets_formula;
    params.ADMIRE_model_space_formulas = formulas_cell;      % Cell that contains the formulas that are used to calculate the ROI and outer model space for ADMIRE

end