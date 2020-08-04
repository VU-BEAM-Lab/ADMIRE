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
        
        
// Description of ccd_double_precision.c: 
// This file contains the MEX-interface and the C code for performing cyclic
// coordinate descent using double precision in order to fit each ADMIRE model
// to its corresponding set of aperture domain frequency data


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mex.h"


// Define the function that calculates the standard deviation of the y_h array, standardizes the y_h array, and calculates the standardized lambda values
void model_fit_preparation(double * y_h, double * residual_y_h, double * standardized_lambda_value, double lambda, double * y_std, int num_observations) {

// Declare and initialize the variable that stores the running sum of y for the fit
double sum_value = 0.0;

// Calculate the running sum for sum_value
for (int observation = 0; observation < num_observations; observation++) {
   double value = y_h[observation];
   sum_value += value;
}

// Calculate the mean of y for the fit
double mean = sum_value / (double)num_observations;

// Declare and initialize the variable that stores the standard deviation of y for the fit
double std = 0.0;

// Calculate the standard deviation of y for the fit
for (int observation = 0; observation < num_observations; observation++) {
   double value_2 = y_h[observation];
   std += ((value_2 - mean) * (value_2 - mean));
}
std = sqrt(std / (double)num_observations);

// Store the standard deviation of y for the fit
*y_std = std; 

// This if statement standardizes lambda and the y data if the standard deviation isn't 0
if (std != (double)0.0) {
   // Calculate and store the standardized lambda value for the fit
   *standardized_lambda_value = lambda / std;
   
   // Standardize y for the fit and store it into the y_h array and the residual_y_h array
   for (int observation = 0; observation < num_observations; observation++) {
       double standardized_value = y_h[observation] / std;
       residual_y_h[observation] = standardized_value;
   }
}
    
}

// Define the function that performs least-squares regression with elastic-net regularization using the cyclic coordinate descent optimization algorithm in order to fit the ADMIRE models to the data
void model_fit(double * B_h, double * X_matrix_h, double * residual_y_h, double y_std, double lambda, int num_observations, int num_predictors, double alpha, double tolerance, int max_iterations) {

// Declare and initialize the variable that stores the maximum weighted (observation weights are all 1 in this case) sum of squares of the changes in the fitted values for one iteration of cyclic coordinate descent
double global_max_change = 1E12;

// Declare and initialize the variable that counts how many iterations of cyclic coordinate descent have been performed
int iteration_count = 0;

// Perform cyclic coordinate descent until either the maximum number of iterations is reached or the maximum weighted (observation weights are all 1 in this case) sum of squares of the changes in the fitted values becomes less than the tolerance
while (global_max_change >= tolerance && iteration_count < max_iterations) {
    // Declare and initialize the variable that stores maximum weighted (observation weights are all 1 in this case) sum of squares of the changes in the fitted values for one iteration of cyclic coordinate descent
    double max_change = 0.0;

    // Declare and initialize the variable that stores the weighted (observation weights are all 1 in this case) sum of squares of the changes in the fitted values that are due to the current predictor coefficient being updated using cyclic coordinate descent
    double change = 0.0;

    // Cycle through all of the predictors for one iteration of cyclic coordinate descent
    for (int j = 0; j < num_predictors; j++) {
        // Obtain the predictor coefficient value for the current predictor
        double B_j = B_h[j];

        // Store the predictor coefficent value before it's updated
        double previous_B_j = B_j;

        // Declare and initialize the variable that stores the correlation between the current predictor column and the residual values that are obtained leaving the current predictor out
        double p_j = 0.0;

        // Calculate the residual values leaving the current predictor out (the predictor coefficients are initialized to zero, so the residual values are going to initially be y)
        // This if-else statement accounts for the fact that the contribution of the current predictor only needs to be removed from the residual values if the predictor coefficient is not zero
        // This is due to the fact that if the predictor coefficient is already zero, then the predictor contribution to the residual is zero
        if (B_j != (double)0.0) {
           for (int observation_row = 0; observation_row < num_observations; observation_row++) {
               // Obtain the correct value from the model matrix for the current predictor
               double X_value = X_matrix_h[(j * num_observations) + observation_row];

               // Remove the contribution of the current predictor from the current residual value
               double residual_y_value = residual_y_h[observation_row] + (X_value * B_j);

               // Store the updated residual value back into the residual_y_h array
               residual_y_h[observation_row] = residual_y_value;

               // Compute the correlation between the current predictor column and the residual values that are obtained leaving the current predictor out 
               // The correlation is computed as a running sum
               p_j = p_j + (X_value * residual_y_value);
           }
        } else {
           for (int observation_row = 0; observation_row < num_observations; observation_row++) {
               // Obtain the correct value from the model matrix for the current predictor
               double X_value = X_matrix_h[(j * num_observations) + observation_row];

               // Obtain the residual value (this is essentially the residual value leaving the current predictor out because the predictor coefficient value is zero) 
               double residual_y_value = residual_y_h[observation_row];

               // Compute the correlation between the current predictor column and the residual values that are obtained leaving the current predictor out
               // The correlation is computed as a running sum
               p_j = p_j + (X_value * residual_y_value);
           }
        } 

        // Divide the computed correlation by the total number of observations in y (also the total number of observations in one predictor column)
        p_j = ((double)1.0 / (double)num_observations) * p_j;

        // Apply the soft-thresholding function that is associated with the L1-regularization component of elastic-net regularization 
        double gamma = lambda * alpha;
        if (p_j > (double)0.0 && gamma < fabs(p_j)) {
           B_j = p_j - gamma;
        } else if (p_j < (double)0.0 && gamma < fabs(p_j)) {
           B_j = p_j + gamma;
        } else {
           B_j = (double)0.0;
        }

        B_j = B_j / (((double)1.0 / (double)num_observations) + (lambda * ((double)1.0 - alpha)));

        // Store the updated predictor coefficient value into the B_h array
        B_h[j] = B_j;

        // Update the residual values to include the contribution of the current predictor using the updated predictor coefficient value 
        // If the updated predictor coefficient value is 0, then it's contribution to the residual values is zero
        if (B_j != (double)0.0) {
           for (int observation_row = 0; observation_row < num_observations; observation_row++) {
               // Store the updated residual back into the residual_y_h array
               residual_y_h[observation_row] = residual_y_h[observation_row] - (X_matrix_h[(j * num_observations) + observation_row] * B_j);
           }
        }

        // Compute the weighted (observation weights are all 1 in this case) sum of squares of the changes in the fitted values (this is used for the tolerance convergence criterion)
        change = ((double)1.0 / (double)num_observations) * ((previous_B_j - B_j) * (previous_B_j - B_j));
        if (change > max_change) {
           max_change = change;
        }
    }

    // Update the global_max_change variable
    global_max_change = max_change;

    // Update the iteration count variable
    iteration_count = iteration_count + 1;
}

// Account for the fact that the y in the model fit was divided by its standard deviation
for (int j = 0; j < num_predictors; j++) {
    B_h[j] = B_h[j] * y_std;
}

}


// Define the MEX gateway function
void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[]) {

// Declare the pointers to the host arrays
double * params_h;
double * X_matrix_h;
double * y_h;

// Obtain the array that contains the parameters
params_h = (double*)mxGetData(prhs[0]);
int num_observations = (int)params_h[0];
int num_predictors = (int)params_h[1];
double alpha = params_h[2];
double lambda = params_h[3];
double tolerance = params_h[4];
int max_iterations = (int)params_h[5];

// Obtain the model matrix
X_matrix_h = (double*)mxGetData(prhs[1]);
   
// Obtain the data that the model matrix is being fitted to
y_h = (double*)mxGetData(prhs[2]);

// Declare and initialize the variable that stores the standard deviation of y
double y_std = 0.0; 

// Declare and initialize the variable that stores the standardized lambda value 
double standardized_lambda_value = 0.0;

// Allocate memory for the residual array
double * residual_y_h = mxMalloc(num_observations * sizeof(double));

// Model fit preparation
model_fit_preparation(y_h, residual_y_h, &standardized_lambda_value, lambda, &y_std, num_observations);

// Assign the pointers to the output arrays
double * B_h;
plhs[0] = mxCreateNumericMatrix(num_predictors, 1, mxDOUBLE_CLASS, mxREAL);
B_h = (double*)mxGetData(plhs[0]);

// Initialize the predictor coefficients to 0
for (int j = 0; j < num_predictors; j++) {
    B_h[j] = (double)0.0;   
}

// Perform the model fit if the standard deviation of y isn't 0
if (y_std != (double)0.0) {
   model_fit(B_h, X_matrix_h, residual_y_h, y_std, standardized_lambda_value, num_observations, num_predictors, alpha, tolerance, max_iterations);
}

// Free the memory allocated for the residual array
mxFree(residual_y_h);

}
