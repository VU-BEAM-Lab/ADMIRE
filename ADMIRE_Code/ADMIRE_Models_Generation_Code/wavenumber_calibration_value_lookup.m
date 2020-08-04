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


% Description of wavenumber_calibration_value_lookup.m:
% This function determines which wavenumber calibration values to use for a
% given transducer center frequency (Hz)


function wavenumber_cal_values = wavenumber_calibration_value_lookup(params)
    
    % Determine the transducer center frequency 
    f0 = params.f0;
    
    % Determine which wavenumber calibration values to use based off of the
    % transducer center frequency 
    wavenumber_cal_values = [1.135 0.995 0.925 0.905];
    if f0 == 1818200
        wavenumber_cal_values = [1.32 0.995 0.835 0.81];
    elseif f0 == 2083300
        wavenumber_cal_values = [1.32 1 0.845 0.825];
    elseif f0 == 2500000
        wavenumber_cal_values = [1.315 0.995 0.835 0.81]; 
    elseif f0 == 3000000
        wavenumber_cal_values = [1.31 0.995 0.835 0.805]; 
    elseif f0 == 3125000
        wavenumber_cal_values = [1.305 0.985 0.825 0.79];
    elseif f0 == 3500000
        wavenumber_cal_values = [1.29 0.99 0.835 0.815];
    elseif f0 == 3600000
        wavenumber_cal_values = [1.315 1.01 0.855 0.815];
    elseif f0 == 4000000
        wavenumber_cal_values = [1.29 0.99 0.845 0.82]; 
    elseif f0 == 4500000
        wavenumber_cal_values = [1.325 1.025 0.87 0.825];
    elseif f0 == 5000000
        wavenumber_cal_values = [1.29 0.995 0.85 0.835];
    elseif f0 == 5208000
        wavenumber_cal_values = [1.145 0.995 0.895 0.865];
    elseif f0 == 5500000
        wavenumber_cal_values = [1.28 0.995 0.855 0.845]; 
    elseif f0 == 6500000
        wavenumber_cal_values = [1.31 1.04 0.89 0.84];
    elseif f0 == 7500000
        wavenumber_cal_values = [1.275 0.975 0.835 0.82];
    elseif f0 == 7800000
        wavenumber_cal_values = [1.295 1 0.855 0.825];
    elseif f0 == 7813000
        wavenumber_cal_values = [1.295 1 0.855 0.825];
    end
            
end