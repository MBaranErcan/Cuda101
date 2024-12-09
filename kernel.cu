/*
These example works here are intended to implement the book "CUDA by Example" by Jason Sanders and Edward Kandrot.
*/

#include <stdio.h>

#include <cuda_runtime.h>

#include <Windows.h>

using namespace std;


bool isKeyPressed(unsigned timeout_ms);
void printCudaDeviceProperties();


int main() {
	
	// Print the Cuda device properties
	printCudaDeviceProperties();




	// Program end
	printf("Press any key to escape...");
	while (!isKeyPressed(0)); // Wait for 10 seconds.

	return 0;
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