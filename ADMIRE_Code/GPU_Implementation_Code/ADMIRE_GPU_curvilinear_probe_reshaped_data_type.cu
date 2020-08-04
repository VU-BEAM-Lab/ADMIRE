// Copyright 2020 Christopher Khan

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the license at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and 
// limitations under the License.
        
        
// Description of ADMIRE_GPU_curvilinear_probe_reshaped_data_type.cu: 
// This file contains the MEX-interface that calls the C/CUDA code for 
// performing ADMIRE on a GPU. It is used when params.data_type = 'Reshaped'
// and params.probe_type = 'Curvilinear'.
        

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cufft.h>
#include "definitions.h"
#include "GPU_processing_kernels.cu"
#include "mex.h"


// Parameters
static int initialized = 0;                      // Specifies whether everything has been initialized or not
static int t0;                                   // Verasonics t0 index (any depth samples before this will be removed), and one is subtracted from it in the GPU code to obtain zero-based indexing
static int num_depths;                           // Number of depth sample
static int num_elements;                         // Number of receive elements used to obtain one beam
static int num_beams;                            // Number of beams
static int start_depth_offset;                   // Index of the first depth sample to which ADMIRE is applied, and one is subtracted from it in the GPU code to obtain zero-based indexing
static int stft_num_zeros;                       // Number of zeros to use for the FFT zero padding when performing the STFT;
static int stft_num_windows;                     // Number of STFT windows for one beam
static int stft_window_shift;                    // Number of depth samples to shift by when moving to the next STFT window
static int stft_length;                          // STFT window length without zero-padding
static int max_windows_per_set;                  // Number of windows to group together for STFT calculation (largest value for this is the number of threads per block divided by the padded STFT length)
static int num_selected_freqs;                   // Number of frequencies within one STFT window to perform ADMIRE on
static int num_corresponding_negative_freqs;     // The number of selected frequencies in one STFT window that have corresponding negative frequencies
static int num_fits;                             // Total number of model fits that are performed
static int total_num_cropped_y_observations;     // Total number of values in the cropped_y_d array
static int total_num_X_matrix_values;            // Total number of values in the X_matrix_d array
static int total_num_B_values;                   // Total number of values in the B_d array
static float alpha;                              // The alpha to use for elastic-net regularization 
static float tolerance;                          // Maximum weighted (observation weights are all 1 in this case) sum of squares of the changes in the fitted values between iterations of cyclic coordinate descent (convergence criterion)
static int max_iterations;                       // Maximum number of cyclic coordinate descent iterations to perform (convergence criterion)
static float lambda_scaling_factor;              // Scaling factor used for the calculation of lambda, which is used in elastic-net regularization
static int scan_conversion_parameters_length;    // Length of each scan conversion parameter vector
static int scan_converted_num_axial_positions;   // Number of depths in the scan-converted image
static int scan_converted_num_lateral_positions; // Number of lateral positions in the scan-converted image
static int channel_data_output_flag;             // Flag that specifies whether or not to output the processed channel data in addition to the envelope data (0 means don't output, and 1 means output) 
        
// GPU device arrays
static float * reshaped_d;                    // Stores the reshaped channel data 
static cudaArray * cuArray;                   // CUDA array for texture memory binding          
static float * delays_d;                      // Stores the calculated delays in samples
static float * delayed_data_d;                // Stores the delayed channel data
static float * stft_window_d;                 // Stores the windowing function coefficients that are used in calculating the STFT of the delayed channel data
static cufftComplex * stft_d;                 // Stores the STFT data
static float * selected_freq_inds_d;          // Stores the indices of the selected frequencies within one STFT window (these indices use zero-based indexing)
static float * negative_freq_inds_d;          // Stores the indices of the negative frequencies that correspond to the positive frequencies being fitted (these indices use zero-based indexing)
static float * negative_freq_include_d;       // Stores the binary flag (0 or 1) that indicates whether a frequency being fitted has a corresponding negative frequency to store the conjugate for
static float * y_d;                           // Stores the standardized STFT data for the selected frequencies (the real components are stacked on top of the imaginary components)
static float * cropped_y_d;                   // Stores the cropped y data that results from applying aperture growth
static float * residual_y_d;                  // Stores the residual values that are obtained during each fit
static float * y_include_mask_d;              // Stores the binary flag (0 or 1) that indicates whether to crop a y value or not
static float * y_std_d;                       // Stores the standard deviations for each portion of the y_d array (one portion corresponds to one elastic net regression fit)
static float * standardized_lambda_d;         // Stores the standardized lambda values for each portion of the y_d array (one portion corresponds to one elastic net regression fit)
static double * num_observations_d;           // Stores the number of observations for each fit after cropping the y data
static double * observation_thread_stride_d;  // Stores the indices corresponding to where each fit starts in the cropped_y_d array (these indices use zero-based indexing)
static double * num_predictors_d;             // Stores the number of predictors for each fit after cropping the y data
static float * X_matrix_d;                    // Stores all of the ADMIRE models matrices 
static double * X_matrix_thread_stride_d;     // Stores the indices corresponding to where each model begins in the X_matrix_d array (these indices use zero-based indexing)
static float * B_d;                           // Stores the predictor coefficient values that are obtained from each fit
static double * B_thread_stride_d;            // Stores the indices corresponding to where each set of predictor coefficients begins in the B_d array (these indices use zero-based indexing)
static float * model_fit_flag_d;              // Stores the flag that determines whether to perform a model fit or not for each model
static cufftComplex * summed_data_d;          // Stores the summed channel data
static float * envelope_d;                    // Stores the envelope data
static float * envelope_max_value_d;          // Stores the maximum value for the envelope data
static float * normalized_log_compressed_envelope_d;  // Stores the normalized and log compressed envelope data
static float * dr_d;                                  // Scan conversion parameter array 
static float * dth_d;                                 // Scan conversion parameter array
static float * i00_d;                                 // Scan conversion parameter array (these indices use zero-based indexing)
static float * i01_d;                                 // Scan conversion parameter array (these indices use zero-based indexing)
static float * i10_d;                                 // Scan conversion parameter array (these indices use zero-based indexing)
static float * i11_d;                                 // Scan conversion parameter array (these indices use zero-based indexing)
static float * idx_d;                                 // Scan conversion parameter array (these indices use zero-based indexing)
static float * envelope_min_value_d;                  // Stores the minimum value for the normalized and log compressed envelope data (need this when doing scan conversion)
static float * row_column_replicated_envelope_d;      // Same as normalized_log_compressed_envelope_d but has the last row and column replicated once (need this when doing scan conversion)
static float * scan_converted_envelope_d;             // Stores the scan-converted envelope data        
        
// Channel format description for cudaArray
static cudaChannelFormatDesc channelDescFLOAT;

// Define a texture 
static texture<float, 2, cudaReadModeElementType> texRef;

// cufft plans
static cufftHandle FFTplan1;   // Handle to the cufft plan that is used to perform the Fourier Transform of each column of each STFT window
static cufftHandle FFTplan2;   // Handle to the cufft plan that is used to perform the Fourier Transform of each column of the summed channel data

        
// Define the kernel that delays the channel data based off of the calculated delays in sample shifts
__global__ void delay_data(float * delayed_data_d, float * delays_d, int t0, int num_depths, int num_elements) {

// Obtain the depth, element, and beam indices
int depth_ind = blockIdx.x;
int elem_ind = threadIdx.x;
int beam_ind = blockIdx.y;

// Obtain the index of the delay
int delay_ind = (beam_ind * num_elements * num_depths) + (elem_ind * num_depths) + depth_ind;

// Obtain the index to store the delayed data
int store_ind = (beam_ind * num_elements * num_depths) + (elem_ind * num_depths) + depth_ind;

// Obtain the index for the column of depth samples (this is technically the row in texture memory because the data is stored in row-major order)
int column = (beam_ind * num_elements) + elem_ind;

// Obtain the delay value (this delay also accounts for t0)
float delay = delays_d[delay_ind] + (float)(t0 - 1);

// Interpolate the data based off of the delay and store the result 
delayed_data_d[store_ind] = tex2D(texRef, delay + 0.5f, (float)column + 0.5f);

}

// Define the function that frees allocated memory on the GPU when the MEX interface is exited
void cleanup() {

mexPrintf("MEX-file is terminating, destroying the arrays\n");

// Free the GPU device arrays
cudaFree(reshaped_d);
cudaFree(delays_d);
cudaFree(delayed_data_d);
cudaFree(stft_window_d);
cudaFree(stft_d);
cudaFree(selected_freq_inds_d);
cudaFree(negative_freq_inds_d);
cudaFree(negative_freq_include_d);
cudaFree(y_d);
cudaFree(cropped_y_d);
cudaFree(residual_y_d);
cudaFree(y_include_mask_d);
cudaFree(y_std_d);
cudaFree(standardized_lambda_d);
cudaFree(num_observations_d);
cudaFree(observation_thread_stride_d);
cudaFree(num_predictors_d);
cudaFree(X_matrix_thread_stride_d);
cudaFree(X_matrix_d);
cudaFree(B_d);
cudaFree(B_thread_stride_d);
cudaFree(model_fit_flag_d);
cudaFree(summed_data_d);
cudaFree(envelope_d);
cudaFree(envelope_max_value_d);
cudaFree(normalized_log_compressed_envelope_d);
cudaFree(dr_d);
cudaFree(dth_d);
cudaFree(i00_d); 
cudaFree(i01_d);  
cudaFree(i10_d);
cudaFree(i11_d);
cudaFree(idx_d);
cudaFree(envelope_min_value_d);
cudaFree(row_column_replicated_envelope_d);
cudaFree(scan_converted_envelope_d);
cudaFreeArray(cuArray);

// Free the cufft plans
cufftDestroy(FFTplan1);
cufftDestroy(FFTplan2);

// Reset the GPU device (need this for profiling the MEX file using the Nvidia Visual Profiler)
cudaDeviceReset();

}

// Define the MEX gateway function
void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[]) {

// Initialize everything if it is the first call to the MEX-file
if (!initialized) {

   // Print to the console
   mexPrintf("MEX-file initializing\n");

   // Define the host arrays
   double * GPU_fixed_params_h;
   float * delays_h;
   float * stft_window_h;
   float * selected_freq_inds_h;
   float * negative_freq_inds_h;
   float * negative_freq_include_h;
   float * y_include_mask_h;
   double * num_observations_h;
   double * observation_thread_stride_h;
   double * num_predictors_h;
   double * X_matrix_thread_stride_h;
   float * X_matrix_h;
   double * B_thread_stride_h;
   float * dr_h;
   float * dth_h;
   float * i00_h;
   float * i01_h;
   float * i10_h;
   float * i11_h;
   float * idx_h;
    
   // Obtain the array that contains the GPU parameters that are fixed
   GPU_fixed_params_h = (double*)mxGetData(prhs[0]);
   t0 = (int)GPU_fixed_params_h[0];
   num_depths = (int)GPU_fixed_params_h[1];
   num_elements = (int)GPU_fixed_params_h[2];
   num_beams = (int)GPU_fixed_params_h[3];
   start_depth_offset = (int)GPU_fixed_params_h[4];
   stft_num_zeros = (int)GPU_fixed_params_h[5];
   stft_num_windows = (int)GPU_fixed_params_h[6];
   stft_window_shift = (int)GPU_fixed_params_h[7];
   stft_length = (int)GPU_fixed_params_h[8];
   max_windows_per_set = (int)GPU_fixed_params_h[9];
   num_selected_freqs = (int)GPU_fixed_params_h[10];
   num_corresponding_negative_freqs = (int)GPU_fixed_params_h[11];
   num_fits = (int)GPU_fixed_params_h[12];
   total_num_cropped_y_observations = (int)GPU_fixed_params_h[13];
   total_num_X_matrix_values = (int)GPU_fixed_params_h[14];
   total_num_B_values = (int)GPU_fixed_params_h[15];
   scan_conversion_parameters_length = (int)GPU_fixed_params_h[16];
   scan_converted_num_axial_positions = (int)GPU_fixed_params_h[17];
   scan_converted_num_lateral_positions = (int)GPU_fixed_params_h[18];
   channel_data_output_flag = (int)GPU_fixed_params_h[19];

   // Obtain the other input arrays
   delays_h = (float*)mxGetData(prhs[1]);
   stft_window_h = (float*)mxGetData(prhs[2]);
   selected_freq_inds_h = (float*)mxGetData(prhs[3]);
   negative_freq_inds_h = (float*)mxGetData(prhs[4]);
   negative_freq_include_h = (float*)mxGetData(prhs[5]);
   y_include_mask_h = (float*)mxGetData(prhs[6]);
   num_observations_h = (double*)mxGetData(prhs[7]);
   observation_thread_stride_h = (double*)mxGetData(prhs[8]);
   num_predictors_h = (double*)mxGetData(prhs[9]);
   X_matrix_thread_stride_h = (double*)mxGetData(prhs[10]);
   X_matrix_h = (float*)mxGetData(prhs[11]);
   B_thread_stride_h = (double*)mxGetData(prhs[12]);
   dr_h = (float*)mxGetData(prhs[13]);
   dth_h = (float*)mxGetData(prhs[14]);
   i00_h = (float*)mxGetData(prhs[15]);
   i01_h = (float*)mxGetData(prhs[16]);
   i10_h = (float*)mxGetData(prhs[17]);
   i11_h = (float*)mxGetData(prhs[18]);
   idx_h = (float*)mxGetData(prhs[19]);

   // Allocate the GPU device arrays
   cudaMalloc(&reshaped_d, (num_depths + t0 - 1) * num_elements * num_beams * sizeof(float));
   cudaMalloc(&delays_d, num_depths * num_elements * num_beams * sizeof(float));
   cudaMalloc(&delayed_data_d, num_depths * num_elements * num_beams * sizeof(float));
   cudaMalloc(&stft_window_d, stft_length * sizeof(float));
   cudaMalloc(&stft_d, stft_num_windows * (stft_length + stft_num_zeros) * num_elements * num_beams * sizeof(cufftComplex));
   cudaMalloc(&selected_freq_inds_d, num_selected_freqs * sizeof(float));
   cudaMalloc(&negative_freq_inds_d, num_corresponding_negative_freqs * sizeof(float));
   cudaMalloc(&negative_freq_include_d, num_selected_freqs * sizeof(float));
   cudaMalloc(&y_d, 2 * stft_num_windows * num_selected_freqs * num_elements * num_beams * sizeof(float));
   cudaMalloc(&cropped_y_d, total_num_cropped_y_observations * sizeof(float));
   cudaMalloc(&residual_y_d, total_num_cropped_y_observations * sizeof(float));
   cudaMalloc(&y_include_mask_d, 2 * num_elements * num_selected_freqs * stft_num_windows * num_beams * sizeof(float));
   cudaMalloc(&y_std_d, num_fits * sizeof(float));
   cudaMalloc(&standardized_lambda_d, num_fits * sizeof(float));
   cudaMalloc(&num_observations_d, num_fits * sizeof(double));
   cudaMalloc(&observation_thread_stride_d, num_fits * sizeof(double));
   cudaMalloc(&num_predictors_d, num_fits * sizeof(double));
   cudaMalloc(&X_matrix_thread_stride_d, num_fits * sizeof(double));
   cudaMalloc(&X_matrix_d, total_num_X_matrix_values * sizeof(float));
   cudaMalloc(&B_d, total_num_B_values * sizeof(float));
   cudaMalloc(&B_thread_stride_d, num_fits * sizeof(double));
   cudaMalloc(&model_fit_flag_d, num_fits * sizeof(float));
   cudaMalloc(&summed_data_d, num_depths * num_beams * sizeof(cufftComplex));
   cudaMalloc(&envelope_d, num_depths * num_beams * sizeof(float));
   cudaMalloc(&envelope_max_value_d, 1 * sizeof(float));
   cudaMalloc(&normalized_log_compressed_envelope_d, num_depths * num_beams * sizeof(float));
   cudaMalloc(&dr_d, scan_conversion_parameters_length * sizeof(float));
   cudaMalloc(&dth_d, scan_conversion_parameters_length * sizeof(float));
   cudaMalloc(&i00_d, scan_conversion_parameters_length * sizeof(float));
   cudaMalloc(&i01_d, scan_conversion_parameters_length * sizeof(float));
   cudaMalloc(&i10_d, scan_conversion_parameters_length * sizeof(float));
   cudaMalloc(&i11_d, scan_conversion_parameters_length * sizeof(float));
   cudaMalloc(&idx_d, scan_conversion_parameters_length * sizeof(float));
   cudaMalloc(&envelope_min_value_d, 1 * sizeof(float));
   cudaMalloc(&row_column_replicated_envelope_d, (num_depths + 1) * (num_beams + 1) * sizeof(float));
   cudaMalloc(&scan_converted_envelope_d, scan_converted_num_axial_positions * scan_converted_num_lateral_positions * sizeof(float));

   // Transfer the data from the host arrays to the GPU device arrays
   cudaMemcpy(delays_d, delays_h, num_depths * num_elements * num_beams * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(stft_window_d, stft_window_h, stft_length * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(selected_freq_inds_d, selected_freq_inds_h, num_selected_freqs * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(negative_freq_inds_d, negative_freq_inds_h, num_corresponding_negative_freqs * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(negative_freq_include_d, negative_freq_include_h, num_selected_freqs * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(y_include_mask_d, y_include_mask_h, 2 * num_elements * num_selected_freqs * stft_num_windows * num_beams * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(num_observations_d, num_observations_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(observation_thread_stride_d, observation_thread_stride_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(num_predictors_d, num_predictors_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(X_matrix_thread_stride_d, X_matrix_thread_stride_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(X_matrix_d, X_matrix_h, total_num_X_matrix_values * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(B_thread_stride_d, B_thread_stride_h, num_fits * sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(dr_d, dr_h, scan_conversion_parameters_length * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(dth_d, dth_h, scan_conversion_parameters_length * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(i00_d, i00_h, scan_conversion_parameters_length * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(i01_d, i01_h, scan_conversion_parameters_length * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(i10_d, i10_h, scan_conversion_parameters_length * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(i11_d, i11_h, scan_conversion_parameters_length * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(idx_d, idx_h, scan_conversion_parameters_length * sizeof(float), cudaMemcpyHostToDevice);

   // Allocate the CUDA array for texture memory
   channelDescFLOAT = cudaCreateChannelDesc<float>();
   cudaMallocArray(&cuArray, &channelDescFLOAT, num_depths + t0 - 1, num_elements * num_beams);

   // Create a cufft plan to take the fast Fourier transform of each column of each STFT window
   cufftPlan1d(&FFTplan1, stft_length + stft_num_zeros, CUFFT_C2C, stft_num_windows * num_elements * num_beams);

   // Create a cufft plan to take the fast Fourier transform of each column of the summed channel data
   cufftPlan1d(&FFTplan2, num_depths, CUFFT_C2C, num_beams);

   // Run the cleanup function when exiting the MEX interface
   mexAtExit(cleanup);

   // Set initialization variable to 1 because everything has been initialized
   initialized = 1;

}




//// THIS SECTION DELAYS THE CHANNEL DATA ////

// Obtain the array that contains the adjustable GPU parameters
float * GPU_adjustable_params_h;
GPU_adjustable_params_h = (float*)mxGetData(prhs[20]);
alpha = GPU_adjustable_params_h[0];
tolerance = GPU_adjustable_params_h[1];
max_iterations = (int)GPU_adjustable_params_h[2];
lambda_scaling_factor = GPU_adjustable_params_h[3];

// Obtain the input data
float * data_h;
data_h = (float*)mxGetData(prhs[21]);

// Set the predictor coefficient values to 0
cudaMemset(B_d, 0, total_num_B_values * sizeof(float));

// Set the model fit flag values to 0
cudaMemset(model_fit_flag_d, 0, num_fits * sizeof(float));

// Transfer the channel data from the host array to the GPU device array
cudaMemcpy(reshaped_d, data_h, (num_depths + t0 - 1) * num_elements * num_beams * sizeof(float), cudaMemcpyHostToDevice);

// Set up texture memory for performing linear interpolation in order to delay the channel data
cudaMemcpyToArray(cuArray, 0, 0, reshaped_d, (num_depths + t0 - 1) * num_elements * num_beams * sizeof(float), cudaMemcpyDeviceToDevice);
texRef.addressMode[0] = cudaAddressModeBorder;
texRef.addressMode[1] = cudaAddressModeBorder;
texRef.filterMode = cudaFilterModeLinear;
texRef.normalized = false;
cudaBindTextureToArray(texRef, cuArray, channelDescFLOAT);

// Define the grid and block dimensions for the delay_data GPU kernel
dim3 DELAY_GRID_SIZE;
DELAY_GRID_SIZE = dim3(num_depths, num_beams, 1);
dim3 DELAY_BLOCK_SIZE;
DELAY_BLOCK_SIZE = dim3(num_elements, 1, 1);

// Call the delay_data GPU kernel in order to delay the channel data by performing linear interpolation
delay_data<<<DELAY_GRID_SIZE, DELAY_BLOCK_SIZE>>>(delayed_data_d, delays_d, t0, num_depths, num_elements);

//// END OF DELAY SECTION ////




//// THIS SECTION CALCULATES THE SHORT-TIME FOURIER TRANSFORM OF THE DELAYED CHANNEL DATA ////

// Set max_windows_per_set to the number of STFT windows if the the number of STFT windows is less than max_windows_per_set
int num_windows_per_set;
if (stft_num_windows < max_windows_per_set) {
   num_windows_per_set = 1;
}  else {
   num_windows_per_set = max_windows_per_set;
}

// Calculate the number of STFT window groupings for the STFT calculation
int num_sets = (int)(ceilf((float)stft_num_windows / (float)num_windows_per_set));

// Calculate the number of STFT windows in the last grouping
int num_windows_per_set_last = stft_num_windows - (num_windows_per_set * (num_sets - 1));

// Obtain the index that corresponds to the last STFT window grouping set
int last_set_ind = num_sets - 1;

// Calculate the zero-padded STFT window length
int stft_padded_length = stft_length + stft_num_zeros; 

// Define the grid and block dimensions for the the stft_preparation GPU kernel
dim3 STFT_PREPARATION_GRID_SIZE;
STFT_PREPARATION_GRID_SIZE = dim3(num_sets, num_elements, num_beams);
dim3 STFT_PREPARATION_BLOCK_SIZE;
STFT_PREPARATION_BLOCK_SIZE = dim3(stft_padded_length, num_windows_per_set, 1); 

// Call the stft_preparation GPU kernel in order to arrange the data for all of the STFT windows and to apply the STFT windowing function coefficients
stft_preparation<<<STFT_PREPARATION_GRID_SIZE, STFT_PREPARATION_BLOCK_SIZE, stft_length * sizeof(float)>>>(stft_d, delayed_data_d, stft_window_d, stft_num_zeros, stft_num_windows, stft_window_shift, stft_length, num_windows_per_set, num_windows_per_set_last, last_set_ind, num_depths, num_elements, start_depth_offset);

// Calculate the short-time Fourier transform of the data by taking the fast Fourier transform of each column within each STFT window
cufftExecC2C(FFTplan1, stft_d, stft_d, CUFFT_FORWARD);

//// END OF SHORT TIME FOURIER TRANSFORM SECTION ////




//// THIS SECTION OBTAINS THE FREQUENCY DATA THAT CORRESPONDS TO THE SELECTED FREQUENCIES AND PROCESSES IT USING ADMIRE ////

// Define the grid and block dimensions for the frequency_selection GPU kernel
dim3 FREQUENCY_SELECTION_GRID_SIZE;
FREQUENCY_SELECTION_GRID_SIZE = dim3(stft_num_windows, num_beams, 1);
dim3 FREQUENCY_SELECTION_BLOCK_SIZE;
FREQUENCY_SELECTION_BLOCK_SIZE = dim3(num_elements, 1, 1);

// Call the frequency_selection GPU kernel in order to obtain the frequency data that corresponds to the selected frequencies for ADMIRE 
frequency_selection<<<FREQUENCY_SELECTION_GRID_SIZE, FREQUENCY_SELECTION_BLOCK_SIZE>>>(y_d, selected_freq_inds_d, stft_d, stft_length, stft_num_zeros, stft_num_windows, num_selected_freqs, num_elements);

// Define the number of model fits to perform within one GPU block
int num_threads_per_block = 32; 

// Set num_threads_per_block to num_fits if the total number of model fits is less than the number of model fits per GPU block
if (num_fits < num_threads_per_block) {
   num_threads_per_block = num_fits;
}

// Calculate the number of GPU blocks that are required to perform all of the model fits
int num_blocks = (int)ceilf((float)num_fits / (float)num_threads_per_block);

// Calculate the number of model fits that are performed within the last GPU block
int num_threads_last_block = num_fits - ((num_blocks - 1) * num_threads_per_block);

// Define the grid and block dimensions for the model_fit_preparation GPU kernel
dim3 MODEL_FIT_PREPARATION_GRID_SIZE;
MODEL_FIT_PREPARATION_GRID_SIZE = dim3(num_blocks, 1, 1);
dim3 MODEL_FIT_PREPARATION_BLOCK_SIZE;
MODEL_FIT_PREPARATION_BLOCK_SIZE = dim3(num_threads_per_block, 1, 1);
model_fit_preparation<<<MODEL_FIT_PREPARATION_GRID_SIZE, MODEL_FIT_PREPARATION_BLOCK_SIZE>>>(cropped_y_d, model_fit_flag_d, y_d, residual_y_d, y_include_mask_d, y_std_d, standardized_lambda_d, num_observations_d, observation_thread_stride_d, lambda_scaling_factor, num_elements, num_threads_per_block, num_blocks, num_threads_last_block);

// Calculate the number of blocks that are required to perform the model fits for all of the beams for one frequency and one STFT window depth range
int num_beam_blocks = (int)ceilf((float)num_beams / (float)num_threads_per_block);

// Define the number of blocks that correspond to the selected frequencies
int num_freq_blocks = num_selected_freqs;

// Define the number of blocks that correspond to the STFT windows
int num_window_blocks = stft_num_windows;

// Define the grid and block dimensions for the model_fit_reconstruction GPU kernel
dim3 MODEL_FIT_RECONSTRUCTION_GRID_SIZE;
MODEL_FIT_RECONSTRUCTION_GRID_SIZE = dim3(num_beam_blocks, num_window_blocks, num_freq_blocks);
dim3 MODEL_FIT_RECONSTRUCTION_BLOCK_SIZE;
MODEL_FIT_RECONSTRUCTION_BLOCK_SIZE = dim3(num_threads_per_block, 1, 1);

// Call the model_fit_reconstruction GPU kernel in order to fit the ADMIRE models to the frequency data and calculate the reconstructed frequency data
model_fit_reconstruction<<<MODEL_FIT_RECONSTRUCTION_GRID_SIZE, MODEL_FIT_RECONSTRUCTION_BLOCK_SIZE, num_threads_per_block * 2 * num_elements * sizeof(float)>>>(B_d, B_thread_stride_d, X_matrix_d, X_matrix_thread_stride_d, cropped_y_d, model_fit_flag_d, observation_thread_stride_d, residual_y_d, y_std_d, standardized_lambda_d, num_observations_d, num_predictors_d, alpha, tolerance, max_iterations, num_elements, num_threads_per_block, num_beams, num_selected_freqs, stft_num_windows);

//// END OF DATA SELECTION AND MODEL FIT SECTION ////




//// THIS SECTION CALCULATES THE INVERSE SHORT-TIME FOURIER TRANSFORM OF THE RECONSTRUCTED SHORT-TIME FOURIER TRANSFORM DATA ////

// Set the stft_d array values to 0 (this is to zero out all of the frequencies that were not reconstructed with ADMIRE)
cudaMemset(stft_d, 0, stft_num_windows * num_elements * num_beams * (stft_length + stft_num_zeros) * sizeof(cufftComplex));

// Define the grid and block dimensions for the inverse_stft_preparation GPU kernel
dim3 ISTFT_PREPARATION_GRID_SIZE;
ISTFT_PREPARATION_GRID_SIZE = dim3(stft_num_windows, 1, 1);
dim3 ISTFT_PREPARATION_BLOCK_SIZE;
ISTFT_PREPARATION_BLOCK_SIZE = dim3(num_beams, 1, 1);

// Call the inverse_stft_preparation GPU kernel in order to place the reconstructed STFT data back into the stft_d array
inverse_stft_preparation<<<ISTFT_PREPARATION_GRID_SIZE, ISTFT_PREPARATION_BLOCK_SIZE>>>(cropped_y_d, selected_freq_inds_d, negative_freq_inds_d, negative_freq_include_d, stft_d, observation_thread_stride_d, y_include_mask_d, stft_length, stft_num_zeros, stft_num_windows, num_selected_freqs, num_elements);

// Calculates the inverse short-time Fourier transform (the window overlap is assumed to be 0 for GPU execution, so the inverse fast fourier transform along each column of each STFT window just needs to be calculated in order to obtain the inverse short-time Fourier transform)
cufftExecC2C(FFTplan1, stft_d, stft_d, CUFFT_INVERSE);

//// END OF INVERSE SHORT-TIME FOURIER TRANSFORM SECTION ////




//// THIS SECTION REMOVES THE ZERO-PADDING THAT WAS ADDED FOR THE SHORT-TIME FOURIER TRANSFORM ////

// Define the grid and block dimensions for the stft_data_array_to_delayed_data_array GPU kernel
dim3 TRANSFER_GRID_SIZE;
TRANSFER_GRID_SIZE = dim3(stft_num_windows, num_beams, 1);
dim3 TRANSFER_BLOCK_SIZE;
TRANSFER_BLOCK_SIZE = dim3(num_elements, 1, 1);

// Call the stft_data_array_to_delayed_data_array GPU kernel in order to remove the STFT zero-padding and to store the reconstructed channel data back into the delayed_data_d array
stft_data_array_to_delayed_data_array<<<TRANSFER_GRID_SIZE, TRANSFER_BLOCK_SIZE>>>(delayed_data_d, stft_d, stft_num_windows, stft_length, stft_num_zeros, num_depths, num_elements, start_depth_offset);

//// END OF ZERO-PADDING REMOVAL SECTION ////




//// THIS SECTION SUMS THE CHANNEL DATA AND CALCULATES THE ENVELOPE DATA ////

// Define the grid and block dimensions for the sum_channel_data GPU kernel
dim3 SUM_GRID_SIZE;
SUM_GRID_SIZE = dim3(num_depths, 1, 1);
dim3 SUM_BLOCK_SIZE;
SUM_BLOCK_SIZE = dim3(num_beams, 1, 1);

// Define the grid and block dimensions for the optimized summing GPU kernels
dim3 SUM_OPTIMIZED_GRID_SIZE;
SUM_OPTIMIZED_GRID_SIZE = dim3(num_depths, num_beams, 1);
dim3 SUM_OPTIMIZED_BLOCK_SIZE;
SUM_OPTIMIZED_BLOCK_SIZE = dim3(num_elements, 1);

// Define a variable that stores the number of elements as an unsigned integer
unsigned int num_elements_uint = (unsigned int)num_elements;

// Define a variable that stores the number of elements minus one as an unsigned integer
unsigned int num_elements_minus_one_uint = (unsigned int)(num_elements - 1);

// Determine if the number of elements is a power of two
bool case_1 = num_elements_uint && !(num_elements_uint & (num_elements_uint - ((unsigned int)1)));

// Determine if the number of elements minus one is a power two
bool case_2 = num_elements_minus_one_uint && !(num_elements_minus_one_uint & (num_elements_minus_one_uint - ((unsigned int)1)));

// Sum the delayed channel data (the conditional statements determine which summing GPU kernel to use)
if (case_1 || case_2) {
   if (num_elements % 2 == 0) {
      // Call the sum_channel_data_optimized_even GPU kernel in order to sum the delayed channel data if the number of elements is a power of two and an even number
      sum_channel_data_optimized_even<<<SUM_OPTIMIZED_GRID_SIZE, SUM_OPTIMIZED_BLOCK_SIZE, num_elements * sizeof(float)>>>(summed_data_d, delayed_data_d, num_depths, num_elements);
   }  else {
      // Call the sum_channel_data_optimized_odd GPU kernel in order to sum the delayed channel data if the number of elements minus one is a power of two and if the number of elements is odd
      sum_channel_data_optimized_odd<<<SUM_OPTIMIZED_GRID_SIZE, SUM_OPTIMIZED_BLOCK_SIZE, num_elements * sizeof(float)>>>(summed_data_d, delayed_data_d, num_depths, num_elements);
   }
} else {
  // Call the sum_channel_data GPU kernel in order to sum the delayed channel data for all other cases (this kernel does not use the optimized summing algorithm that is used by the other two kernels)
  sum_channel_data<<<SUM_GRID_SIZE, SUM_BLOCK_SIZE>>>(summed_data_d, delayed_data_d, num_depths, num_elements);
}

// Calculate the fast Fourier transform for each column of the summed channel data that has been beamformed
cufftExecC2C(FFTplan2, summed_data_d, summed_data_d, CUFFT_FORWARD);

// Define the grid and block sizes for the hilbert_weighting GPU kernel
dim3 HILBERT_WEIGHTING_GRID_SIZE;
HILBERT_WEIGHTING_GRID_SIZE = dim3(num_depths, 1, 1);
dim3 HILBERT_WEIGHTING_BLOCK_SIZE;
HILBERT_WEIGHTING_BLOCK_SIZE = dim3(num_beams, 1, 1);

// Call the hilbert_weighting GPU kernel in order to apply weighting to the data according to the Hilbert Transform algorithm that MATLAB uses
hilbert_weighting<<<HILBERT_WEIGHTING_GRID_SIZE, HILBERT_WEIGHTING_BLOCK_SIZE>>>(summed_data_d, num_depths);

// Calculate the inverse fast Fourier transform of each column of the weighted data 
cufftExecC2C(FFTplan2, summed_data_d, summed_data_d, CUFFT_INVERSE);

// Define the grid and block sizes for the envelope GPU kernel
dim3 ENVELOPE_GRID_SIZE;
ENVELOPE_GRID_SIZE = dim3(num_depths, 1, 1);
dim3 ENVELOPE_BLOCK_SIZE;
ENVELOPE_BLOCK_SIZE = dim3(num_beams, 1, 1);

// Call the envelope GPU kernel in order to obtain the envelope data
envelope<<<ENVELOPE_GRID_SIZE, ENVELOPE_BLOCK_SIZE>>>(envelope_d, summed_data_d, num_depths);

//// END OF SUM AND ENVELOPE CALCULATION SECTION ////


        
        
//// THIS SECTION NORMALIZES AND LOG COMPRESSES THE ENVELOPE DATA ////

// Define the grid and block dimensions for the max_envelope_value GPU kernel
dim3 MAX_ENVELOPE_VALUE_GRID_SIZE;
MAX_ENVELOPE_VALUE_GRID_SIZE = dim3(1, 1, 1);
dim3 MAX_ENVELOPE_VALUE_BLOCK_SIZE;
MAX_ENVELOPE_VALUE_BLOCK_SIZE = dim3(num_beams, 1, 1);

// Call the maximum_envelope_value GPU kernel in order to obtain the maximum value of the envelope data
maximum_envelope_value<<<MAX_ENVELOPE_VALUE_GRID_SIZE, MAX_ENVELOPE_VALUE_BLOCK_SIZE, num_beams * sizeof(float)>>>(envelope_max_value_d, envelope_d, num_depths, num_beams);

// Define the grid and block sizes for the envelope_normalization_and_log_compression GPU kernel
dim3 NORMALIZE_LOG_COMPRESS_GRID_SIZE;
NORMALIZE_LOG_COMPRESS_GRID_SIZE = dim3(num_depths, 1, 1);
dim3 NORMALIZE_LOG_COMPRESS_BLOCK_SIZE;
NORMALIZE_LOG_COMPRESS_BLOCK_SIZE = dim3(num_beams, 1, 1);

// Call the envelope_normalization_and_log_compression GPU kernel in order to normalize and apply log compression to the envelope data
envelope_normalization_and_log_compression<<<NORMALIZE_LOG_COMPRESS_GRID_SIZE, NORMALIZE_LOG_COMPRESS_BLOCK_SIZE>>>(normalized_log_compressed_envelope_d, envelope_d, envelope_max_value_d, num_depths);

//// END OF ENVELOPE NORMALIZATION AND LOG COMPRESSION SECTION ////
        



//// THIS SECTION SCAN CONVERTS THE NORMALIZED AND LOG-COMPRESSED ENVELOPE ////

// Define the grid and block sizes for the minimum_envelope_value GPU kernel
dim3 MIN_ENVELOPE_VALUE_GRID_SIZE; 
MIN_ENVELOPE_VALUE_GRID_SIZE = dim3(1, 1, 1);
dim3 MIN_ENVELOPE_VALUE_BLOCK_SIZE; 
MIN_ENVELOPE_VALUE_BLOCK_SIZE = dim3(num_beams, 1, 1);

// Call the minimum_envelope_value GPU kernel in order to obtain the minimum value of the normalized and log-compressed envelope data
minimum_envelope_value<<<MIN_ENVELOPE_VALUE_GRID_SIZE, MIN_ENVELOPE_VALUE_BLOCK_SIZE, num_beams * sizeof(float)>>>(envelope_min_value_d, normalized_log_compressed_envelope_d, num_depths, num_beams);

// Define the grid and block sizes of the row_column_replicate GPU kernel
dim3 REPLICATION_GRID_SIZE;
REPLICATION_GRID_SIZE = dim3(num_depths + 1, 1, 1);
dim3 REPLICATION_BLOCK_SIZE;
REPLICATION_BLOCK_SIZE = dim3(num_beams + 1, 1, 1);

// Call the row_column_replicate GPU kernel in order to replicate the last row and the last column of the normlaized and log-compressed envelope data
row_column_replicate<<<REPLICATION_GRID_SIZE, REPLICATION_BLOCK_SIZE>>>(row_column_replicated_envelope_d, normalized_log_compressed_envelope_d, num_depths, num_beams);

// Calculate the total number of pixels that are in the scan-converted image
int total_num_pixels = scan_converted_num_axial_positions * scan_converted_num_lateral_positions;

// Define the number of threads to use within one block
int num_threads_per_block_scan_convert = 512;

// Set the number of threads per block to the total number of pixels in the scan-converted image if the total number of pixels is less than the number of threads per block
if (total_num_pixels < num_threads_per_block_scan_convert) {
   num_threads_per_block_scan_convert = total_num_pixels;
}

// Calculate the number of blocks that are required to perform the initialization of the scan-converted image
int num_blocks_scan_convert = ceilf((float)total_num_pixels / (float)num_threads_per_block_scan_convert);

// Calculate the number of threads that are used within the last block
int num_threads_last_block_scan_convert = total_num_pixels - ((num_blocks_scan_convert - 1) * num_threads_per_block_scan_convert);

// Define the grid and block dimensions for the scan_converted_envelope_initialization GPU kernel
dim3 SCAN_CONVERT_INIT_GRID_SIZE;
SCAN_CONVERT_INIT_GRID_SIZE = dim3(num_blocks_scan_convert, 1, 1);
dim3 SCAN_CONVERT_INIT_BLOCK_SIZE; 
SCAN_CONVERT_INIT_BLOCK_SIZE = dim3(num_threads_per_block_scan_convert, 1, 1);

// Call the scan_converted_envelope_initialization GPU kernel in order to initialize every pixel of the scan-converted envelope to the minimum value of the normalized and log-compressed envelope data
scan_converted_envelope_initialization<<<SCAN_CONVERT_INIT_GRID_SIZE, SCAN_CONVERT_INIT_BLOCK_SIZE>>>(scan_converted_envelope_d, envelope_min_value_d, num_threads_per_block_scan_convert, num_threads_last_block_scan_convert, num_blocks_scan_convert);

// Define the number of threads to use within one block
int num_threads_per_block_scan_convert_2 = 512;

// Set the number of threads per block to the scan_conversion_parameters_length if scan_conversion_parameters_length is less than the number of threads per block
if (scan_conversion_parameters_length < num_threads_per_block_scan_convert_2) {
   num_threads_per_block_scan_convert_2 = scan_conversion_parameters_length;
}

// Calculate the number of blocks that are required to perform scan conversion
int num_blocks_scan_convert_2 = ceilf((float)scan_conversion_parameters_length / (float)num_threads_per_block_scan_convert_2);

// Calculate the number of threads that are used within the last block
int num_threads_last_block_scan_convert_2 = scan_conversion_parameters_length - ((num_blocks_scan_convert_2 - 1) * num_threads_per_block_scan_convert_2);

// Define the grid and block dimensions for the scan_conversion GPU kernel
dim3 SCAN_CONVERSION_GRID_SIZE;
SCAN_CONVERSION_GRID_SIZE = dim3(num_blocks_scan_convert_2, 1, 1);
dim3 SCAN_CONVERSION_BLOCK_SIZE;
SCAN_CONVERSION_BLOCK_SIZE = dim3(num_threads_per_block_scan_convert_2, 1, 1);

// Call the scan_conversion GPU kernle in order to perform scan conversion of the normalized and log-compressed envelope data
scan_conversion<<<SCAN_CONVERSION_GRID_SIZE, SCAN_CONVERSION_BLOCK_SIZE>>>(scan_converted_envelope_d, row_column_replicated_envelope_d, dr_d, dth_d, idx_d, i00_d, i01_d, i10_d, i11_d, num_threads_per_block_scan_convert_2, num_threads_last_block_scan_convert_2, num_blocks_scan_convert_2); 

//// END OF SCAN CONVERSION SECTION ////


        
        
//// THIS SECTION OBTAINS THE MEX-FILE OUTPUTS AND UNBINDS THE TEXTURE MEMORY ////

// Declare the pointers to the MEX-file outputs
float * scan_converted_envelope_h;
float * delayed_data_h;

// Allocate the scan_converted_envelope_h array (this is the output array for the scan-converted envelope data)
plhs[0] = mxCreateNumericMatrix(scan_converted_num_axial_positions, scan_converted_num_lateral_positions, mxSINGLE_CLASS, mxREAL);
scan_converted_envelope_h = (float*)mxGetData(plhs[0]);

// Allocate the delayed_data_d array if the channel_data_output_flag parameter is set to 1 (this outputs the reconstructed channel data as a column vector that can be reshaped in MATLAB)
if (channel_data_output_flag == 1) {
   plhs[1] = mxCreateNumericMatrix(num_depths * num_elements * num_beams, 1, mxSINGLE_CLASS, mxREAL);
   delayed_data_h = (float*)mxGetData(plhs[1]);
}

// Transfer the scan-converted envelope data
cudaMemcpy(scan_converted_envelope_h, scan_converted_envelope_d, scan_converted_num_axial_positions * scan_converted_num_lateral_positions * sizeof(float), cudaMemcpyDeviceToHost);

// Transfer the reconstructed channel data if the channel_data_output_flag parameter is set to 1
if (channel_data_output_flag == 1) {
   cudaMemcpy(delayed_data_h, delayed_data_d, num_depths * num_elements * num_beams * sizeof(float), cudaMemcpyDeviceToHost);
}

// Unbind the texture memory
cudaUnbindTexture(texRef);

//// END OF OUTPUT AND UNBIND SECTION ////

} 