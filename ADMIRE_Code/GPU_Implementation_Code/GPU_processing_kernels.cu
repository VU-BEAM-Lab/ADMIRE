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
        
        
// Description of GPU_processing_kernels.cu: 
// This file contains the CUDA code for performing ADMIRE on a GPU
        

// Define the GPU kernel that reshapes the Verasonics data and changes its data type from int16 to float
__global__ void reshape_data(float * reshaped_d, int16 * Verasonics_RF_buffer_data_d, int t0, int num_buffer_rows, int num_depths, int num_elements) {

// Obtain the depth, element, and beam indices
int depth_ind = blockIdx.x;
int elem_ind = threadIdx.x;
int beam_ind = blockIdx.y;

// Calculate the global index of the element on the entire aperture (this shifts the aperture when doing a walked aperture)
int global_elem_ind = elem_ind + beam_ind;

// Calculate the index for storing the reshaped data into the reshaped_data_d array
int store_ind = (beam_ind * num_elements * (num_depths + t0 - 1)) + (elem_ind * (num_depths + t0 - 1)) + depth_ind;

// Calculate the index for accessing the data in the Verasonics_RF_buffer_data_d array
int access_ind = (global_elem_ind * num_buffer_rows) + (beam_ind * (num_depths + t0 - 1)) + depth_ind;

// Convert the reshaped data to the float data type and store it into the reshaped_data_d array
reshaped_d[store_ind] = (float)(Verasonics_RF_buffer_data_d[access_ind]); 

}




// Define the GPU kernel for summing the delayed channel data across the elements (this works for any number of elements, but it does not use the optimized summing algorithm)
__global__ void sum_channel_data(cufftComplex * summed_data_d, float * delayed_data_d, int num_depths, int num_elements) {

// Obtain the depth and beam indices
int depth_ind = blockIdx.x;
int beam_ind = threadIdx.x;

// Calculate the index for storing the data into the summed_data_d array
int store_ind = (beam_ind * num_depths) + depth_ind;

// Declare and initialize the variable that will store the sum across the elements for one depth and one beam position
float sum = 0;

// Loop through and sum the data across the elements for one depth and one beam position
for (int elem_ind = 0; elem_ind < num_elements; elem_ind++) {
    sum += delayed_data_d[(beam_ind * num_elements * num_depths) + (elem_ind * num_depths) + depth_ind];
}

// Store the sum for the real component of the summed_data_d array
summed_data_d[store_ind].x = sum;

// Store 0 for the imaginary component of the summed_data_d array (a cufftComplex data type is used for the summed_data_d array because the Fourier Transform of each column is calculated later on)
summed_data_d[store_ind].y = 0.0f;

}




// Define the optimized GPU kernel for summing the delayed channel data across the elements (used when the number of elements is a power of two)
__global__ void sum_channel_data_optimized_even(cufftComplex * summed_data_d, float * delayed_data_d, int num_depths, int num_elements) {

// Define the shared memory array to store the data across the elements
// for one depth and beam position (the amount of bytes is declared in the GPU kernel call)
extern __shared__ float sdata[];

// Obtain the depth, element, and beam indices
int depth_ind = blockIdx.x;
int elem_ind = threadIdx.x;
int beam_ind = blockIdx.y;

// Calculate the index for storing the data into the summed_data_d array
int store_ind = (beam_ind * num_depths) + depth_ind;

// Calculate the index for accessing the delayed channel data in the delayed_data_d array
int access_ind = (beam_ind * num_elements * num_depths) + (elem_ind * num_depths) + depth_ind;

// Store the data into the shared memory array
sdata[elem_ind] = delayed_data_d[access_ind];

// Synchronize all of the threads within a block
__syncthreads();

// Sum the data using a parallel reduction algorithm that was
// obtained from Mark Harris' Nvidia presentation titled
// "Optimizing Parallel Reduction in CUDA"
for (unsigned int s = num_elements / 2; s > 0; s >>= 1) {
    int s_integer = (int)s;
    if (elem_ind < s_integer) {
       sdata[elem_ind] += sdata[elem_ind + s_integer];
    }
    __syncthreads();
}

if (elem_ind == 0) {
   // Store the sum for the real component of the summed_data_d array
   summed_data_d[store_ind].x = sdata[0];

   // Store 0 for the imaginary component of the summed_data_d array (a cufftComplex data type is used for the summed_data_d array because the fast Fourier transforms are calculated later on)
   summed_data_d[store_ind].y = 0.0f;
}

}




// Define the optimized GPU kernel for summing the delayed channel data across the elements (used when the number of elements minus one is a power of two and the number of elements is odd)
__global__ void sum_channel_data_optimized_odd(cufftComplex * summed_data_d, float * delayed_data_d, int num_depths, int num_elements) {

// Define the shared memory array to store the data across the elements
// for one depth and beam position (the amount of bytes is declared in the GPU kernel call)
extern __shared__ float sdata[];

// Obtain the depth, element, and beam indices
int depth_ind = blockIdx.x;
int elem_ind = threadIdx.x;
int beam_ind = blockIdx.y;

// Calculate the index to store the data into the summed_data_d array
int store_ind = (beam_ind * num_depths) + depth_ind; 

// Calculate the index for accessing the delayed channel data in the delayed_data_d array
int access_ind = (beam_ind * num_elements * num_depths) + (elem_ind * num_depths) + depth_ind;

// Store the data into the shared memory array
sdata[elem_ind] = delayed_data_d[access_ind];

// Synchronize all of the threads within a block
__syncthreads();

// Sum the data using a parallel reduction algorithm that was
// obtained from Mark Harris' Nvidia presentation titled
// "Optimizing Parallel Reduction in CUDA"
for (unsigned int s = (num_elements - 1) / 2; s > 0; s >>= 1) {
    int s_integer = (int)s;
    if (elem_ind < s_integer) {
       sdata[elem_ind] += sdata[elem_ind + s_integer];
    }
    __syncthreads();
}

if (elem_ind == 0) {
   // Store the sum for the real component of the summed_data_d array
   summed_data_d[store_ind].x = sdata[0] + sdata[num_elements - 1];

   // Store 0 for the imaginary component of the summed_data_d array (a cufftComplex data type is used for the summed_data_d array because the fast Fourier transforms are calculated later on)
   summed_data_d[store_ind].y = 0.0f;
}

}




// Define the GPU kernel that applies the Hilbert transform weights using the same algorithm that MATLAB uses
__global__ void hilbert_weighting(cufftComplex * summed_data_d, int num_depths) {

// Obtain the depth and beam indices
int depth_ind = blockIdx.x;
int beam_ind = threadIdx.x;

// Calculate the index for storing the data into the summed_data_d array
int store_ind = (beam_ind * num_depths) + depth_ind;

// Apply weights to the frequency data
if (num_depths % 2 == 0) {
   if (depth_ind >= 1 && depth_ind <= ((num_depths / 2) - 1)) {
      summed_data_d[store_ind].x *= 2;
      summed_data_d[store_ind].y *= 2;
   }  else if (depth_ind >= ((num_depths / 2) + 1)) {
      summed_data_d[store_ind].x = 0.0f;
      summed_data_d[store_ind].y = 0.0f;
   }
}  else {
   if (depth_ind >= 1 && depth_ind <= (((num_depths + 1) / 2) - 1)) {
      summed_data_d[store_ind].x *= 2;
      summed_data_d[store_ind].y *= 2;
   }  else if (depth_ind > (((num_depths + 1) / 2) - 1)) {
      summed_data_d[store_ind].x = 0.0f;
      summed_data_d[store_ind].y = 0.0f;       
   }
} 

}




// Define the GPU kernel that finds the magnitude of the Hilbert transform and returns the envelope data
__global__ void envelope(float * envelope_d, cufftComplex * summed_data_d, int num_depths) {

// Obtain the depth and beam indices
int depth_ind = blockIdx.x;
int beam_ind = threadIdx.x;

// Calculate the index for accessing the summed data in the summed_data_d array and storing the envelope data into the envelope_h array
int access_store_ind = (beam_ind * num_depths) + depth_ind;
   
// Obtain the I and Q data values and divide them by the inverse fast Fourier transform length to account for the inverse fast Fourier transform normalization factor
float I = summed_data_d[access_store_ind].x / num_depths;
float Q = summed_data_d[access_store_ind].y / num_depths;

// Calculate the magnitude using I and Q and store the result into the envelope_d array
envelope_d[access_store_ind] = sqrtf((I * I) + (Q * Q));

}




// Define the GPU kernel that prepares the data for the short-time Fourier transform
__global__ void stft_preparation(cufftComplex * stft_d, float * delayed_data_d, float * stft_window_d, int stft_num_zeros, int stft_num_windows, int stft_window_shift, int stft_length, int num_windows_per_set, int num_windows_per_set_last, int last_set_ind, int num_depths, int num_elements, int start_depth_offset) {

// Define the shared memory array to store the STFT windowing function coefficients (the amount of bytes is declared in the GPU kernel call)
extern __shared__ float sdata[];

// Obtain the index of the set that contains multiple windows, and obtain the element, beam, window, and window sample indices
int set_ind = blockIdx.x;
int elem_ind = blockIdx.y;
int beam_ind = blockIdx.z;
int window_ind = threadIdx.y;
int window_sample_ind = threadIdx.x;

// Store the STFT windowing function coefficients into the shared memory array
if (window_sample_ind == 0 && window_ind == 0) {
   for (int i = 0; i < stft_length; i++) {
       sdata[i] = stft_window_d[i];
   }
}

// Synchronize all of the threads within a block
__syncthreads();

// Declare the num_windows_per_set_2 variable
int num_windows_per_set_2;

// This if-else statement accounts for the fact that the last set might have less windows
if (set_ind == last_set_ind) {
   num_windows_per_set_2 = num_windows_per_set_last;
}  else {
   num_windows_per_set_2 = num_windows_per_set;
}

// This if statement makes sure that extra threads aren't doing data processing if the last set has less windows
if (window_ind < num_windows_per_set_2) {
   // Calculate the depth index
   int depth_ind = (set_ind * num_windows_per_set * stft_window_shift) + (window_ind * stft_window_shift) + window_sample_ind + (start_depth_offset - 1);

   // Calculate the index for accessing the delayed channel data in the delayed_data_d array
   int access_ind = (beam_ind * num_elements * num_depths) + (elem_ind * num_depths) + depth_ind;

   // Calculate the index for storing the data into the stft_d array
   int store_ind = (beam_ind * num_elements * stft_num_windows * (stft_length + stft_num_zeros)) + (elem_ind * stft_num_windows * (stft_length + stft_num_zeros)) + (set_ind * num_windows_per_set * (stft_length + stft_num_zeros)) + (window_ind * (stft_length + stft_num_zeros)) + window_sample_ind;

   // This if-else statement handles the zero-padding by adding zeros at the end of each STFT window
   if (window_sample_ind < stft_length) {
      // Obtain the delayed channel data, multiply it by its corresponding STFT windowing function coefficient, and store it for the real component of the stft_d array
      stft_d[store_ind].x = delayed_data_d[access_ind] * sdata[window_sample_ind];

      // Store 0 for the imaginary component of the stft_d array (a cufftComplex data type is used for the stft_d array because the fast Fourier transforms are calculated within the STFT windows later on)
      stft_d[store_ind].y = 0.0f;
   }  else {
      // Add zero-padding to the STFT windows
      stft_d[store_ind].x = 0.0f;
      stft_d[store_ind].y = 0.0f;
   }
}

}




// Define the GPU kernel that obtains the frequency data corresponding to the selected frequencies for ADMIRE
__global__ void frequency_selection(float * y_d, float * selected_freq_inds_d, cufftComplex * stft_d, int stft_length, int stft_num_zeros, int stft_num_windows, int num_selected_freqs, int num_elements) {

// Obtain the element, window, and beam indices along with the frequency number
int elem_ind = threadIdx.x;
int freq_num = blockIdx.x;
int window_ind = blockIdx.y;
int beam_ind = blockIdx.z;

// Calculate the stride to move to the correct position of the stft_d array
int stride = (beam_ind * num_elements * stft_num_windows * (stft_length + stft_num_zeros)) + (elem_ind * stft_num_windows * (stft_length + stft_num_zeros)) + (window_ind * (stft_length + stft_num_zeros));

// Obtain the index of the window sample that corresponds to the frequency
int freq_ind = (int)selected_freq_inds_d[freq_num];

// Calculate the index to store the real component of the STFT data into the y_d array
int store_ind_real = (beam_ind * stft_num_windows * num_selected_freqs * 2 * num_elements) + (window_ind * num_selected_freqs * 2 * num_elements) + (freq_num * 2 * num_elements) + elem_ind;

// Calculate the index to store the imaginary component of the STFT data into the y_d array
int store_ind_imaginary = (beam_ind * stft_num_windows * num_selected_freqs * 2 * num_elements) + (window_ind * num_selected_freqs * 2 * num_elements) + (freq_num * 2 * num_elements) + num_elements + elem_ind;

// Access and store the real component of the STFT data into the y_d array
y_d[store_ind_real] = stft_d[freq_ind + stride].x;

// Access and store the imaginary component of the STFT data into the y_d array
y_d[store_ind_imaginary] = stft_d[freq_ind + stride].y;

}




// Define the GPU kernel that calculates the standard deviations for each portion of the y_d array, standardizes the y_d array, and calculates the standardized lambda values
__global__ void model_fit_preparation(float * cropped_y_d, float * model_fit_flag_d, float * y_d, float * residual_y_d, float * y_include_mask_d, double * start_ind_d, float * y_std_d, float * standardized_lambda_d, double * num_observations_d, double * observation_thread_stride_d, float lambda_scaling_factor, int num_elements, int num_threads_per_block, int num_blocks, int num_threads_last_block) {

// Define the shared memory array to store the aperture domain frequency data for the fits within one block (the amount of bytes is declared in the GPU kernel call)
extern __shared__ float sdata[];

// Obtain the index of the block
int block_ind = blockIdx.x;

// Obtain the thread index within one block
int block_thread_ind = threadIdx.x;

// Calculate the fit index
int fit_ind = (block_ind * num_threads_per_block) + block_thread_ind;

// Determine how many threads are in the block (accounts for the fact that the last block may contain less active threads than the other blocks)
int num_threads_per_block_2 = num_threads_per_block;
if (block_ind == (num_blocks - 1)) {
   num_threads_per_block_2 = num_threads_last_block;
}

// This if statement makes sure that extra threads aren't doing data processing if the last block has less fits to perform
if (block_thread_ind < num_threads_per_block_2) {
   // Calculate the number of observations that the fit would have if aperture growth was not applied
   int no_aperture_growth_num_observations = 2 * num_elements;

   // Obtain the number of observations for the fit
   int num_observations = (int)num_observations_d[fit_ind];

   // Obtain the thread stride that is used to obtain the correct set of observations in the cropped_y_d array for the fit
   int observation_thread_stride = (int)observation_thread_stride_d[fit_ind];

   // Declare and initialize the variable that stores the running sum of y for the fit
   float sum_value = 0.0f;

   // Declare and initialize the variable that stores the running sum of y squared for the fit
   float y_dot_product_value = 0.0f;

   // Declare and initialize the variable that keeps track of where to store each observation (need this because the y data is being cropped)
   int count = 0;

   // Calculate the running sums for sum_value and y_dot_product_value
   for (int observation = 0; observation < no_aperture_growth_num_observations; observation++) {
       // Declare the variable to determine whether an observation is included or not due to aperture growth
       int include_flag = (int)y_include_mask_d[observation + (fit_ind * no_aperture_growth_num_observations)];
       
       // Store the selected y value into shared memory
       if (include_flag == 1) {
          float value = y_d[observation + (fit_ind * no_aperture_growth_num_observations)];
          sum_value += value;
          y_dot_product_value += (value * value);
          sdata[(count * num_threads_per_block) + block_thread_ind] = value;

          // Store the index of the first location where the binary aperture growth mask has a 1 for the fit
          if (count == 0) {
             start_ind_d[fit_ind] = (double)observation;
          }
          count = count + 1;
       }
   }

   // Calculate the mean of y for the fit
   float mean = sum_value / (float)num_observations;
   
   // Declare and initialize the variable that stores the standard deviation of y for the fit
   float std = 0.0f;

   // Calculate the standard deviation of y for the fit
   for (int observation = 0; observation < num_observations; observation++) {
       float value_2 = sdata[(observation * num_threads_per_block) + block_thread_ind];
       std += ((value_2 - mean) * (value_2 - mean));
   }
   std = sqrtf(std / (float)num_observations);

   // Store the standard deviation of y for the fit in the y_std_d array
   y_std_d[fit_ind] = std;

   // This if statement standardizes the lambda value and y only if the standard deviation of y is not equal to 0
   if (std != 0.0f) {
      // Store 1 for the model fit flag in order to indicate that a model fit should be performed for this model
      model_fit_flag_d[fit_ind] = 1.0f;

      // Calculate the standardized lambda value and store it into the standardized_lambda_d array 
      standardized_lambda_d[fit_ind] = (lambda_scaling_factor * sqrtf(y_dot_product_value / (float)num_observations)) / std;

      // Standardize y for the fit and store it into the cropped_y_d array and the residual_y_d array
      for (int observation = 0; observation < num_observations; observation++) {
          residual_y_d[observation_thread_stride + observation] = sdata[(observation * num_threads_per_block) + block_thread_ind] / std;
      }
   }
}

}  




// Define the GPU kernel that performs least-squares regression with elastic-net regularization using the cyclic coordinate descent optimization algorithm in order to fit the ADMIRE models to the data
__global__ void model_fit_reconstruction(float * B_d, double * B_thread_stride_d, float * X_matrix_d, double * X_matrix_thread_stride_d, float * cropped_y_d, float * model_fit_flag_d, double * observation_thread_stride_d, float * residual_y_d, float * y_std_d, float * standardized_lambda_d, double * num_observations_d, double * num_predictors_d, float alpha, float tolerance, int max_iterations, int num_elements, int num_threads_per_block, int num_beams, int num_selected_freqs, int stft_num_windows) {

// Define the shared memory array to store the residual values of the fits within one block (the amount of bytes is declared in the GPU kernel call)
extern __shared__ float sdata[];

// Obtain the beam block, window, and frequency indices
int beam_block_ind = blockIdx.x; 
int window_ind = blockIdx.y;
int selected_freq_ind = blockIdx.z;

// Obtain the thread index within one block
int block_thread_ind = threadIdx.x;

// Calculate the beam index across the beam blocks for a single frequency and window
int beam_ind = (beam_block_ind * num_threads_per_block) + block_thread_ind;

// Calculate the fit index
int fit_ind = (beam_ind * num_selected_freqs * stft_num_windows) + (window_ind * num_selected_freqs) + selected_freq_ind;

// This if statement makes sure that extra threads aren't doing data processing if the last beam block has less threads
if (beam_ind < num_beams) { 
   // Obtain the thread stride that is used to obtain the correct set of predictors for the fit
   int predictor_thread_stride = (int)B_thread_stride_d[fit_ind];

   // Obtain the thread stride that is used to obtain the correct ADMIRE model matrix for the fit
   int X_thread_stride = (int)X_matrix_thread_stride_d[fit_ind];

   // Obtain the thread stride that is used to obtain the correct set of observations for the fit
   int observation_thread_stride = (int)observation_thread_stride_d[fit_ind];

   // Obtain the standardized lambda value for the fit
   float lambda = standardized_lambda_d[fit_ind];

   // Obtain the number of observations for the fit
   int num_observations = (int)num_observations_d[fit_ind];

   // Obtain the number of predictors for the fit
   int num_predictors = (int)num_predictors_d[fit_ind];

   // Obtain the flag that determines whether to perform a model fit or not
   int model_fit_flag = (int)model_fit_flag_d[fit_ind];

   // This if statement makes sure that a model fit is performed only if the model fit flag is equal to 1 
   if (model_fit_flag == 1) {
      // Declare and initialize the variable that stores the maximum weighted (observation weights are all 1 in this case) sum of squares of the changes of the fitted values for one iteration of cyclic coordinate descent
      float global_max_change = 1E12;

      // Declare and initialize the variable that counts how many iterations of cyclic coordinate descent have been performed
      int iteration_count = 0;
  
      // Store the residual values for the fit into the shared memory array  
      for (int observation_row = 0; observation_row < num_observations; observation_row++) {
          int store_ind = (observation_row * num_threads_per_block) + block_thread_ind;
          sdata[store_ind] = residual_y_d[observation_thread_stride + observation_row];
      }

      // Perform cyclic coordinate descent until either the maximum number of iterations is reached or the maximum weighted (observation weights are all 1 in this case) sum of squares of the changes in the fitted values becomes less than the tolerance
      while (global_max_change >= tolerance && iteration_count < max_iterations) {
            // Declare and initialize the variable that stores the maximum weighted (observation weights are all 1 in this case) sum of squares of the changes in the fitted values for one iteration of cyclic coordinate descent
            float max_change = 0.0f;

            // Declare and initialize the variable that stores the weighted (observation weights are all 1 in this case) sum of squares of the changes in the fitted values that are due to the current predictor coefficient being updated using cyclic coordinate descent
            float change = 0.0f;
         
            // Cycle through all of the predictors for one iteration of cyclic coordinate descent
            for (int j = 0; j < num_predictors; j++) {
                // Obtain the predictor coefficient value for the current predictor
                float B_j = B_d[predictor_thread_stride + j];

                // Store the predictor coefficent value before it's updated
                float previous_B_j = B_j;
       
                // Declare and initialize the variable that stores the correlation between the current predictor column and the residual values that are obtained leaving the current predictor out
                float p_j = 0.0f;

                // Calculate the residual values leaving the current predictor out (the predictor coefficients are initialized to zero, so the residual values are going to initially be y)
                // This if-else statement accounts for the fact that the contribution of the current predictor only needs to be removed from the residual values if the predictor coefficient is not zero
                // This is due to the fact that if the predictor coefficient is already zero, then the predictor contribution to the residual is zero
                if (B_j != 0.0f) {
                   for (int observation_row = 0; observation_row < num_observations; observation_row++) {
                       // Obtain the correct value from the ADMIRE model matrix for the current predictor
                       float X_value = X_matrix_d[X_thread_stride + (j * num_observations) + observation_row];
 
                       // Remove the contribution of the current predictor from the current residual value
                       float residual_y_value = sdata[(observation_row * num_threads_per_block) + block_thread_ind] + (X_value * B_j);

                       // Store the updated residual value back into the shared memory array
                       sdata[(observation_row * num_threads_per_block) + block_thread_ind] = residual_y_value;
      
                       // Compute the correlation between the current predictor column and the residual values that are obtained leaving the current predictor out 
                       // The correlation is computed as a running sum
                       p_j = p_j + (X_value * residual_y_value);
                   }
                } else {
                   for (int observation_row = 0; observation_row < num_observations; observation_row++) {
                       // Obtain the correct value from the ADMIRE model matrix for the current predictor
                       float X_value = X_matrix_d[X_thread_stride + (j * num_observations) + observation_row];

                       // Obtain the residual value (this is essentially the residual value leaving the current predictor out because the predictor coefficient value is zero) 
                       float residual_y_value = sdata[(observation_row * num_threads_per_block) + block_thread_ind];

                       // Compute the correlation between the current predictor column and the residual values that are obtained leaving the current predictor out
                       // The correlation is computed as a running sum
                       p_j = p_j + (X_value * residual_y_value);
                   }
                } 

                // Divide the computed correlation by the total number of observations in y (also the total number of observations in one predictor column)
                p_j = (1.0f / (float)num_observations) * p_j;

                // Apply the soft-thresholding function that is associated with the L1-regularization component of elastic-net regularization 
                float gamma = lambda * alpha;
                if (p_j > 0.0f && gamma < fabsf(p_j)) {
                   B_j = p_j - gamma;
                } else if (p_j < 0.0f && gamma < fabsf(p_j)) {
                   B_j = p_j + gamma;
                } else {
                   B_j = 0.0f;
                }

                // Calculate the updated predictor coefficient value by applying the component of elastic-net regularization that is associated with L2-regularization 
                // The 1/N term comes from the derivation of the coordinate descent update for a predictor coefficient
                B_j = B_j / ((1.0f / (float)num_observations) + (lambda * (1.0f - alpha)));

                // Store the updated predictor coefficient value into the B_d array
                B_d[predictor_thread_stride + j] = B_j;

                // Update the residual values to include the contribution of the current predictor using the updated predictor coefficient value 
                // If the updated predictor coefficient value is 0, then its contribution to the residual values is zero
                if (B_j != 0.0f) {
                   for (int observation_row = 0; observation_row < num_observations; observation_row++) {
                       sdata[(observation_row * num_threads_per_block) + block_thread_ind] = sdata[(observation_row * num_threads_per_block) + block_thread_ind] - (X_matrix_d[X_thread_stride + (j * num_observations) + observation_row] * B_j);
                   }
                }

                // Compute the weighted (observation weights are all 1 in this case) sum of squares of the changes in the fitted values (this is used for the tolerance convergence criterion)
                change = (1.0f / (float)num_observations) * ((previous_B_j - B_j) * (previous_B_j - B_j));
                if (change > max_change) {
                   max_change = change;
                }
            }
   
            // Update the global_max_change variable
            global_max_change = max_change;
        
            // Update the iteration count variable
            iteration_count = iteration_count + 1;
      }
 

      // Unstandardize the estimated predictor coefficient values
      float std_y = y_std_d[fit_ind];
      for (int j = 0; j < num_predictors; j++) {
          B_d[predictor_thread_stride + j] = B_d[predictor_thread_stride + j] * std_y;
      }
   }

   // Store 0 into the relevant entries of the shared memory array in order to use it for storing the reconstructed aperture domain frequency data instead of the residual values
   for (int observation_row = 0; observation_row < num_observations; observation_row++) {
       sdata[(observation_row * num_threads_per_block) + block_thread_ind] = 0.0f;          
   }

   // Reconstruct y using only the estimated predictor coefficient values that correspond to the ROI 
   // The ROI predictors are in the first fourth and third fourth of the each ADMIRE model matrix because the real and complex components are tiled 
   // These two nested for loops are doing X_ROI * B_ROI
   for (int predictor_column = 0; predictor_column < ((int)(num_predictors / 4)); predictor_column++) {
       int predictor_column_2 = predictor_column + ((int)(num_predictors / 2));
       for (int observation_row = 0; observation_row < num_observations; observation_row++) {
           int store_ind = (observation_row * num_threads_per_block) + block_thread_ind;
           sdata[store_ind] = sdata[store_ind] + (X_matrix_d[X_thread_stride + (predictor_column * num_observations) + observation_row] * B_d[predictor_thread_stride + predictor_column]) + (X_matrix_d[X_thread_stride + (predictor_column_2 * num_observations) + observation_row] * B_d[predictor_thread_stride + predictor_column_2]);
       }
   }

   // Store the reconstructed value of y in the cropped_y_d array
   for (int observation_row = 0; observation_row < num_observations; observation_row++) {
       cropped_y_d[observation_thread_stride + observation_row] = sdata[(observation_row * num_threads_per_block) + block_thread_ind];   
   }
}

}




// Define the GPU kernel that places the reconstructed STFT data back into the stft_d array
__global__ void inverse_stft_preparation(float * cropped_y_d, float * selected_freq_inds_d, float * negative_freq_inds_d, float * negative_freq_include_d, cufftComplex * stft_d, double * observation_thread_stride_d, float * y_include_mask_d, double * start_ind_d, double * num_observations_d, int stft_length, int stft_num_zeros, int stft_num_windows, int num_selected_freqs, int num_elements) {

// Obtain the element, window, and beam indices along with the frequency number
int freq_num = blockIdx.x;        
int window_ind = blockIdx.y;
int beam_ind = blockIdx.z;
int elem_ind = threadIdx.x;

// Calculate the number of observations that each fit would have if aperture growth was not applied
int no_aperture_growth_num_observations = 2 * num_elements;

// Calculate the fit index
int fit_ind = (beam_ind * stft_num_windows * num_selected_freqs) + (window_ind * num_selected_freqs) + freq_num;

// Obtain the index of the first location in the binary aperture growth mask that contains a 1 for the fit
int start_ind = (int)start_ind_d[fit_ind];

// Obtain the number of observations for the fit
int num_observations = (int)num_observations_d[fit_ind];

// Calculate the number of observations divided by 2 for the fit
int half_num_observations = (int)(num_observations / 2);

// Obtain the thread stride that is used to obtain the correct set of observations from the cropped_y_d array for the fit
int observation_thread_stride = (int)observation_thread_stride_d[fit_ind];

// Obtain the index of the window sample that corresponds to the frequency
int freq_ind = (int)selected_freq_inds_d[freq_num];

// Obtain the negative frequency flag value to determine if the current frequency has a corresponding negative frequency 
// This is for storing the conjugate of the positive frequency's reconstructed signal for the negative frequency
int negative_freq_flag = (int)negative_freq_include_d[freq_num];

// Obtain the negative frequency flag for the first frequency in order to determine whether the first frequency is 0 or not
int first_freq_flag = (int)negative_freq_include_d[0];

// Initialize the variable that stores index of the window sample that corresponds to the negative frequency
// This variable is not used if the current frequency is 0 because 0 does not have a corresponding negative frequency
int negative_freq_ind = 0;

// Handle the case where there is a corresponding negative frequency
if (negative_freq_flag == 1) {
   // This if statement accounts for the fact that the first frequency that is fit might be 0 instead of a positive frequency that has a corresponding negative frequency
   if (first_freq_flag == 1) {
      // Obtain the index of the window sample that corresponds to the negative frequency
      negative_freq_ind = (int)negative_freq_inds_d[freq_num];
   } else {
      // Obtain the index of the window sample that corresponds to the negative frequency
      negative_freq_ind = (int)negative_freq_inds_d[freq_num - 1];   
   }

   // Calculate the stride to move to the correct position of the stft_d array
   int stride = (beam_ind * num_elements * stft_num_windows * (stft_length + stft_num_zeros)) + (elem_ind * stft_num_windows * (stft_length + stft_num_zeros)) + (window_ind * (stft_length + stft_num_zeros));
 
   // Obtain the include flag value to determine whether an observation is included or not due to aperture growth
   int include_flag = (int)y_include_mask_d[(fit_ind * no_aperture_growth_num_observations) + elem_ind];

   // Store the real and the imaginary components of the reconstructed signal into the stft_d array (also store the conjugate for the corresponding negative frequency)
   if (include_flag == 1) {
      float value = cropped_y_d[observation_thread_stride + elem_ind - start_ind];
      float value_2 = cropped_y_d[observation_thread_stride + half_num_observations + elem_ind - start_ind];
      stft_d[stride + freq_ind].x = value;
      stft_d[stride + freq_ind].y = value_2;
      stft_d[stride + negative_freq_ind].x = value;
      stft_d[stride + negative_freq_ind].y = -1.0f * value_2;
   }
       
} else {
   // Handle the case where there isn't a corresponding negative frequency
   // Calculate the stride to move to the correct position of the stft_d array
   int stride = (beam_ind * num_elements * stft_num_windows * (stft_length + stft_num_zeros)) + (elem_ind * stft_num_windows * (stft_length + stft_num_zeros)) + (window_ind * (stft_length + stft_num_zeros));

   // Obtain the include flag value to determine whether an observation is included or not due to aperture growth
   int include_flag = (int)y_include_mask_d[(fit_ind * no_aperture_growth_num_observations) + elem_ind];

   // Store the real and imaginary components of the reconstructed signal into the stft_d array
   if (include_flag == 1) {
      stft_d[stride + freq_ind].x = cropped_y_d[observation_thread_stride + elem_ind - start_ind];
      stft_d[stride + freq_ind].y = cropped_y_d[observation_thread_stride + half_num_observations + elem_ind - start_ind];
   }
} 

} 




// Define the GPU kernel that places the ISTFT data into the delayed_data_d array (the zero-padding used for the STFT is also removed)
__global__ void stft_data_array_to_delayed_data_array(float * delayed_data_d, cufftComplex * stft_d, int stft_num_windows, int stft_length, int stft_num_zeros, int num_depths, int num_elements, int start_depth_offset) {

// Obtain the window, beam, and element indices
int window_sample_ind = blockIdx.x;
int window_ind = blockIdx.y;
int beam_ind = blockIdx.z;
int elem_ind = threadIdx.x;

// Calculate the stride to move to the correct position of the stft_d array
int stride = (beam_ind * num_elements * stft_num_windows * (stft_length + stft_num_zeros)) + (elem_ind * stft_num_windows * (stft_length + stft_num_zeros)) + (window_ind * (stft_length + stft_num_zeros));

// Calculate the depth index
int depth_ind = (window_ind * stft_length) + window_sample_ind + (start_depth_offset - 1);

// Obtain the index to store the data into the delayed_data_d array
int store_ind = (beam_ind * num_elements * num_depths) + (elem_ind * num_depths) + depth_ind;

// Divide the real component of the data sample from the stft_d array by the inverse fast Fourier transform length to account for the inverse fast Fourier transform normalization factor
// Store the scaled real component into the delayed_data_d array
delayed_data_d[store_ind] = stft_d[stride + window_sample_ind].x / ((float)(stft_length + stft_num_zeros));

}  




// Define the GPU kernel that finds the maximum value of the envelope data
__global__ void maximum_envelope_value(float * envelope_max_value_d, float * envelope_d, int num_depths, int num_beams) {

// Define the shared memory array that stores the maximum envelope data value for each beam (the amount of bytes is declared in the GPU kernel call)
extern __shared__ float sdata[];

// Obtain the beam index
int beam_ind = threadIdx.x;

// Declare and initialize the variable that stores the maximum value for the beam
float beam_max_value = 0.0f;

// Find the maximum envelope data value for each beam and store it into the shared memory array
for (int depth_ind = 0; depth_ind < num_depths; depth_ind++) {
    // Calculate the index to access data from the envelope_d array
    int access_ind = (beam_ind * num_depths) + depth_ind;

    // Obtain the envelope data value and see if it is greater than the current maximum value for the beam
    float value = envelope_d[access_ind];
    if (value > beam_max_value) {
       beam_max_value = value;
    }
}

// Store the maximum value for the beam into the shared memory array
sdata[beam_ind] = beam_max_value;

// Synchronize all of the threads within the block
__syncthreads(); 

// Find the maximum value across all of the beams
if (beam_ind == 0) {
   // Declare and initialize the variable that stores the maximum envelope data value across all of the beams
   float maximum_envelope_value = 0.0f;

   // Loop through the beam maximum values and find the maximum value across all of the beams
   for (int i = 0; i < num_beams; i++) {
       // Obtain the maximum value for the current beam
       float beam_max_value = sdata[i];

       // See if the maximum value for the current beam is greater than the current maximum value for the envelope data
       if (beam_max_value > maximum_envelope_value) {
          maximum_envelope_value = beam_max_value;
       }
   }

   // Store the maximum value across all of the beams into the envelope_max_value_d array
   envelope_max_value_d[0] = maximum_envelope_value;
}

}




// Define the GPU kernel that normalizes and log compresses the envelope data
__global__ void envelope_normalization_and_log_compression(float * normalized_log_compressed_envelope_d, float * envelope_d, float * envelope_max_value_d, int num_depths) {

// Obtain the depth and beam indices
int depth_ind = blockIdx.x;
int beam_ind = threadIdx.x;
   
// Obtain the maximum envelope value across all of the beams
float maximum_envelope_value = envelope_max_value_d[0];

// Calculate the index to access data from the envelope_d array and store the normalized and log-compressed data into the normalized_log_compressed_envelope_d array
int access_store_ind = (beam_ind * num_depths) + depth_ind;

// Normalize, log compress, and store the data value into the normalized_log_compressed_envelope_d array
normalized_log_compressed_envelope_d[access_store_ind] = 20.0f * log10f((envelope_d[access_store_ind] / maximum_envelope_value));

}




// Define the GPU kernel that finds the minimum value of the normalized and log-compressed envelope data
__global__ void minimum_envelope_value(float * envelope_min_value_d, float * normalized_log_compressed_envelope_d, int num_depths, int num_beams) {

// Define the shared memory array that stores the minimum normalized and log-compressed envelope data value for each beam (the amount of bytes is declared in the GPU kernel call)
extern __shared__ float sdata[];

// Obtain the beam index
int beam_ind = threadIdx.x;

// Declare and initialize the variable that stores the minimum value for the beam
float beam_min_value = 0.0f;

// Find the minimum normalized and log-compressed envelope data value for each beam and store it into the shared memory array
for (int depth_ind = 0; depth_ind < num_depths; depth_ind++) {
    // Calculate the index to access data from the normalized_log_compressed_envelope_d array
    int access_ind = (beam_ind * num_depths) + depth_ind;

    // Obtain the normalized and log-compressed envelope data value and see if it is less than the current minimum value for the beam
    float value = normalized_log_compressed_envelope_d[access_ind];
    if (value < beam_min_value) {
       beam_min_value = value;
    }
}

// Store the minimum normalized and log-compressed envelope data value for the beam into shared memory
sdata[beam_ind] = beam_min_value;

// Synchronize all of the threads within the block
__syncthreads(); 

// Find the minimum normalized and log-compressed envelope data value across all of the beams
if (beam_ind == 0) {
   // Declare and initialize the variable that stores the minimum normalized and log-compressed envelope data value across all of the beams
   float minimum_envelope_value = 0.0f;

   // Loop through the beam minimum values and find the minimum value across all of the beams
   for (int i = 0; i < num_beams; i++) {
       // Obtain the minimum value for the current beam
       float beam_min_value = sdata[i];

       // See if the minimum value for the current beam is less than the current minimum for the normalized and log-compressed envelope data
       if (beam_min_value < minimum_envelope_value) {
          minimum_envelope_value = beam_min_value;
       }
   }

   // Store the minimum value across all of the beams into the envelope_min_value_d array
   envelope_min_value_d[0] = minimum_envelope_value;
}

}




// Define the GPU kernel that replicates the last row and the last column of the normalized_log_compressed_envelope_data_d array
__global__ void row_column_replicate(float * row_column_replicated_envelope_d, float * normalized_log_compressed_envelope_d, int num_depths, int num_beams) {

// Obtain the depth and beam indices
int depth_ind = blockIdx.x;
int beam_ind = threadIdx.x;

// Account for the replication cases
// Assign the depth index to the depth_ind_2 variable
int depth_ind_2 = depth_ind;

// Assign the index of the last depth in the normalized_log_compressed_envelope_d array to the index corresponding to the replicated row
if (depth_ind_2 == num_depths) {  
   depth_ind_2 = num_depths - 1;
}

// Assign the beam index to the beam_ind_2 variable
int beam_ind_2 = beam_ind;

// Assign the index of the last beam in the normalized_log_compressed_envelope_d array to the index corresponding to the replicated column
if (beam_ind_2 == num_beams) {
   beam_ind_2 = num_beams - 1;
}

// Calculate the index to access data from the normalized_log_compressed_envelope_d array
int access_ind = (beam_ind_2 * num_depths) + depth_ind_2;
    
// Calculate the index to store data into the row_colum_replicated_envelope_d array
int store_ind = (beam_ind * (num_depths + 1)) + depth_ind;

// Access data from the normalized_log_compressed_envelope_d array and store it into the row_column_replicated_envelope_d array
row_column_replicated_envelope_d[store_ind] = normalized_log_compressed_envelope_d[access_ind];

}




// Define the GPU kernel that initializes the scan_converted_envelope_d array to have the minimum value of the normalized_log_compressed_envelope_d array
__global__ void scan_converted_envelope_initialization(float * scan_converted_envelope_d, float * envelope_min_value_d, int num_threads_per_block_scan_convert, int num_threads_last_block_scan_convert, int num_blocks_scan_convert) {

// Obtain the block index
int block_ind = blockIdx.x;

// Obtain the thread index within one block
int block_thread_ind = threadIdx.x;

// Calculate the global thread index
int global_ind = (block_ind * num_threads_per_block_scan_convert) + block_thread_ind;

// Determine how many threads are in the block (accounts for the fact that the last block may contain less active threads than the other blocks)
int num_threads = num_threads_per_block_scan_convert;
if (block_ind == (num_blocks_scan_convert - 1)) {
   num_threads = num_threads_last_block_scan_convert;
}

// This if statement makes sure that extra threads aren't doing data processing if the last block has less active threads
if (block_thread_ind < num_threads) {
   // Obtain the minimum value of the normalized_log_compressed_envelope_d array
   float minimum_value = envelope_min_value_d[0];

   // Store the minimum value of the normalized_log_compressed_envelope_d array into the scan_converted_envelope_d array
   scan_converted_envelope_d[global_ind] = minimum_value;
}

}




// Define the GPU kernel that performs scan conversion of the normalized and log-compressed envelope data
__global__ void scan_conversion(float * scan_converted_envelope_d, float * row_column_replicated_envelope_d, float * dr_d, float * dth_d, float * idx_d, float * i00_d, float * i01_d, float * i10_d, float * i11_d, int num_threads_per_block_scan_convert_2, int num_threads_last_block_scan_convert_2, int num_blocks_scan_convert_2) {

// Obtain the block index
int block_ind = blockIdx.x;

// Obtain the thread index within one block
int block_thread_ind = threadIdx.x;

// Calculate the global thread index
int global_ind = (block_ind * num_threads_per_block_scan_convert_2) + block_thread_ind;

// Determine how many threads are in the block (accounts for the fact that the last block may contain less active threads than the other blocks)
int num_threads = num_threads_per_block_scan_convert_2;
if (block_ind == (num_blocks_scan_convert_2 - 1)) {
   num_threads = num_threads_last_block_scan_convert_2;
}

// This if statement makes sure that extra threads aren't doing data processing if the last block has less active threads
if (block_thread_ind < num_threads) {
   // Obtain the value of dr_d for the thread
   float dr_value = dr_d[global_ind];

   // Obtain the value of dth_d for the thread
   float dth_value = dth_d[global_ind];

   // Obtain the index of i00_d for the thread
   int i00_ind = (int)i00_d[global_ind];

   // Obtain the index of i01_d for the thread
   int i01_ind = (int)i01_d[global_ind];

   // Obtain the index of i10_d for the thread
   int i10_ind = (int)i10_d[global_ind];

   // Obtain the index of i11_d for the thread
   int i11_ind = (int)i11_d[global_ind];

   // Obtain the index of idx_d for the thread
   int idx_ind = (int)idx_d[global_ind];

   // Obtain the value corresponding to i00_d for the thread
   float i00_value = row_column_replicated_envelope_d[i00_ind];

   // Obtain the value corresponding to i01_d for the thread
   float i01_value = row_column_replicated_envelope_d[i01_ind];

   // Obtain the value corresponding to i10_d for the thread
   float i10_value = row_column_replicated_envelope_d[i10_ind];

   // Obtain the value corresponding to i11_d for the thread
   float i11_value = row_column_replicated_envelope_d[i11_ind];

   // Perform bilinear interpolation and store the result into the scan_converted_envelope_d array
   scan_converted_envelope_d[idx_ind] = ((i10_value - i00_value) * dr_value) + ((i01_value - i00_value) * dth_value) + ((i11_value + i00_value - i10_value - i01_value) * dth_value * dr_value) + i00_value;
}

}   