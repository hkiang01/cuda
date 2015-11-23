// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024  //@@ You can change this
#define SCAN_SIZE (2*BLOCK_SIZE)

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
	    }                                                                          \
    } while (0)

__global__ void helper(float auxVal, float *output, float len)
{
	if (threadIdx.x<len) {
		output[threadIdx.x] += auxVal;
	}
}

__global__ void scan(float *input, float *output, int len) {
	//@@ Modify the body of this function to complete the functionality of
	//@@ the scan on the device
	//@@ You may need multiple kernel calls; write your kernels before this
	//@@ function and call them from here

	__shared__ float partialScan[2 * BLOCK_SIZE];

	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i<len)
		partialScan[threadIdx.x] = input[i];


	//work efficient reduction phase
	for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
		unsigned int index = (threadIdx.x + 1)*stride * 2 - 1;
		if (index<SCAN_SIZE && index < len)
			partialScan[index] += partialScan[index - stride];
		__syncthreads();
	}

	//work efficient post reduction phase
	for (unsigned int stride = BLOCK_SIZE / 2; stride>0; stride /= 2) {
		__syncthreads();
		unsigned int index = (threadIdx.x + 1)*stride * 2 - 1;
		if (index + stride < SCAN_SIZE && index + stride < len) {
			partialScan[index + stride] += partialScan[index];
		}
	}

	__syncthreads();
	if (i<SCAN_SIZE && i < len)
		output[i] += partialScan[threadIdx.x];
}

int main(int argc, char **argv) {
	wbArg_t args;
	float *hostInput;  // The input 1D list
	float *hostOutput; // The output list
	float *deviceInput;
	float *deviceOutput;
	int numElements; // number of elements in the list

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
	hostOutput = (float *)malloc(numElements * sizeof(float));
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The number of input elements in the input is ", numElements);

	wbTime_start(GPU, "Allocating GPU memory.");
	wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Clearing output memory.");
	wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
	wbTime_stop(GPU, "Clearing output memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
		cudaMemcpyHostToDevice));
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	//@@ Initialize the grid and block dimensions here
	dim3 dimGrid(((numElements - 1) / SCAN_SIZE) + 1);
	dim3 dimBlock(BLOCK_SIZE);
	printf("SCAN_SIZE: %d\n", SCAN_SIZE);
	printf("dimGrid: %d\n", dimGrid.x);
	printf("dimBlock: %d\n", dimBlock.x);

	wbTime_start(Compute, "Performing CUDA computation");
	//@@ Modify this to complete the functionality of the scan
	//@@ on the deivce

	//single block
	int i = 0;
	for (int remaining = numElements; remaining > 0; remaining -= SCAN_SIZE) {
		int offset = i*SCAN_SIZE;
		scan << <dimGrid, dimBlock >> >(&deviceInput[offset], &deviceOutput[offset], remaining);
		i++;
	}

	/*
	float aux[dimGrid.x];
	int iter = 0;
	for(int remaining = numElements; remaining > 0; remaining -= SCAN_SIZE) {
	printf("for iteration\n");
	scan<<<dimGrid, dimBlock>>>(&deviceInput[iter*SCAN_SIZE], &deviceOutput[iter*SCAN_SIZE], remaining);

	int lastSize = remaining;
	if(SCAN_SIZE < lastSize)
	lastSize = SCAN_SIZE;


	aux[iter] = deviceOutput[iter*SCAN_SIZE+lastSize-1];
	iter++;
	}

	int remaining = numElements;
	for(int i = 0; i < dimGrid.x-1 && remaining>0; i++)
	{
	int local_len = remaining;
	if(SCAN_SIZE < local_len)
	local_len = SCAN_SIZE;
	//write to scan blocks i+1 to dimGrid.x
	helper<<<dimGrid,dimBlock>>>(aux[i], &deviceOutput[(i+1)*SCAN_SIZE], local_len);
	remaining -= SCAN_SIZE;
	}
	*/

	/*
	int remaining = numElements;
	float intermediate;
	for(int iter = 0; 2*iter < dimGrid.x; iter++) {
	int in_len = numElements;
	if(2*dimBlock.x<in_len)
	in_len = 2*dimBlock.x;
	scan<<<dimGrid,dimBlock>>>(&deviceInput[iter*2*dimBlock.x], &deviceOutput[iter*dimBlock.x], in_len);
	remaining -= 2*dimBlock.x;

	if(iter+1 < dimGrid.x-1) {
	intermediate = deviceOutput[(iter+1)*2*dimBlock.x-1];
	printf("intermediate: %f\n", intermediate);
	in_len = remaining;
	if(2*dimBlock.x < in_len)
	in_len = 2*dimBlock.x;
	helper<<<dimGrid,dimBlock>>>(&deviceInput[(iter+1)*2*dimBlock.x], &deviceOutput[(iter+1)*2*dimBlock.x], in_len, intermediate);
	}
	}
	*/
	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
		cudaMemcpyDeviceToHost));
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	cudaFree(deviceInput);
	cudaFree(deviceOutput);
	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, hostOutput, numElements);

	free(hostInput);
	free(hostOutput);

	return 0;
}