/*
 * Code originally from https://devblogs.nvidia.com/even-easier-introduction-cuda/
 */

#include <iostream>
#include <math.h>

// function to add the elements of two arrays
__global__ // this makes it run on the gpu
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

__global__
void add_threads(int n, float*x, float*y)
{
	int index = threadIdx.x;
	int stride = blockDim.x;
	for (int i=index; i<n; i+=stride)
	{
		y[i] = x[i] + y[i];
	}
}

int main(void)
{
  int N = 1<<20; // 1M elements

  //float *x = new float[N];
  //float *y = new float[N];
  // Allocating "Unified Memory" -- can be accessed from both cpu and gpu
  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  //add(N, x, y);
  // Launch it on one gpu thread like this:
  //add<<<1, 1>>>(N, x, y);
  add_threads<<<1, 255>>>(N, x, y);

  // And we need to wait for the gpu to finish
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  //delete [] x;
  //delete [] y;
  cudaFree(x);
  cudaFree(y);

  return 0;
}
