﻿/*
These example works here are intended to implement the book "CUDA by Example" by Jason Sanders and Edward Kandrot.
*/

#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <Windows.h>
#include <chrono>

using namespace std;

#define N 10000

// Function prototypes
bool isKeyPressed(unsigned timeout_ms);
void printCudaDeviceProperties();

__global__ void add(int* a, int* b, int* c);

/* ----------MAIN--------- */
int main(void) {
	// Program start
	
	// start timer
	auto start = chrono::high_resolution_clock::now();


	// Print the Cuda device properties
	printCudaDeviceProperties();

	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	// Allocate memory on the device.
	cudaMalloc((void**) &dev_a, N * sizeof(int));
	cudaMalloc((void**) &dev_b, N * sizeof(int));
	cudaMalloc((void**) &dev_c, N * sizeof(int));


	// Fill the arrays a and b
	for (int i = 0; i < N; i++) {
		a[i] = 2-i;
		b[i] = i * i;
	}

	// Copy the arrays to device
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);


	//add<<<1, 1>>>(dev_a, dev_b, dev_c); // No parallelism?
	//add<<<N, 1>>>(dev_a, dev_b, dev_c); // Use N blocks, 1 thread per block
	//add<<<1, N>>>(dev_a, dev_b, dev_c); // Use 1 block, N threads per block
	add<<<(N + 127)/128, 128 >>> (dev_a, dev_b, dev_c); // Use ceil(N/128) blocks, 128 threads per block


	// Copy the result back to host
	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);


	// Print the result , commented out bc it takes too long to print.
	//for (int i = 0; i < N; i++) {
	//	printf("a[%d]:%d + b[%d]:%d = c[%d]:%d\n", i, a[i], i, b[i], i, c[i]);
	//}

	// Free the memory
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);


	// end timer
	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed = end - start;
	printf("Time taken: %f seconds\n", elapsed.count());

	// Program end
	printf("Press any key to escape...");
	while (!isKeyPressed(0)); // Wait for 10 seconds.

	return 0;
}


// Kernel function to add two arrays
__global__ void add(int* a, int* b, int* c) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
	printf("Thread %d finished\n", tid);
}

// Check if a key is pressed, with a default timeout of 0 ms. 
// If key is pressed return true, if not wait auntil timeout and send false. 
// This is windows specific.
bool isKeyPressed(unsigned timeout_ms = 0)
{
	return WaitForSingleObject(
		GetStdHandle(STD_INPUT_HANDLE),
		timeout_ms
	) == WAIT_OBJECT_0;
}


// Print the device properties
void printCudaDeviceProperties() {
	cudaDeviceProp prop;

	int count, i;
	cudaGetDeviceCount(&count);

	for (i = 0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("Device name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);
		printf("Device copy overlap: ");
		if (prop.deviceOverlap)
			printf("Enabled\n");
		else
			printf("Disabled\n");
		printf("Kernel execution timeout: ");
		if (prop.kernelExecTimeoutEnabled)
			printf("Enabled\n");
		else
			printf("Disabled\n");

		printf("   --- Memory Information for device %d ---\n", i);
		printf("Total global mem: %Iu\n", prop.totalGlobalMem); // Hint: to print size_t correctly, use %Iu, in windows.
		printf("Total constant mem: %Iu\n", prop.totalConstMem);
		printf("Max mem pitch: %Iu\n", prop.memPitch);
		printf("Texture Alignment: %Iu\n", prop.textureAlignment);

		printf("   --- MP Information for device: %d ---\n", i);
		printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
		printf("Shared mem per mp: %Iu\n", prop.sharedMemPerBlock);
		printf("Registers per mp: %d\n", prop.regsPerBlock);
		printf("Threads in warp: %d\n", prop.warpSize);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("\n");
	}
}