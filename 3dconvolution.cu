#include    <wb.h>

#define wbCheck(stmt) do {                                         \
        cudaError_t err = stmt;                                    \
        if (err != cudaSuccess) {                                  \
            wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err)); \
            wbLog(ERROR, "Failed to run stmt ", #stmt);            \
            return -1;                                             \
		        }                                                          \
	    } while(0)


//@@ Define any useful program-wide constants here
#define KERNEL_WIDTH 3
#define KERNEL_RADIUS ((KERNEL_WIDTH - 1)/2)
#define O_TILE_WIDTH 8
#define BLOCK_WIDTH (O_TILE_WIDTH + KERNEL_WIDTH - 1)
//#define BLOCK_WIDTH 5
//@@ Define constant memory for device kernel here
__constant__ float M[KERNEL_WIDTH*KERNEL_WIDTH*KERNEL_WIDTH];

__global__ void conv3d(float *A, float *B,
	const int z_size, const int y_size, const int x_size) {
	//@@ Insert kernel code here
	int x_out = blockIdx.x*blockDim.x + threadIdx.x;
	int y_out = blockIdx.y*blockDim.y + threadIdx.y;
	int z_out = blockIdx.z*blockDim.z + threadIdx.z;

	//the kernel is cached, no need to load it

	//If (z_out, y_out, x_out) is within bounds, do more work
	if (x_out >= 0 && x_out < x_size &&
		y_out >= 0 && y_out < y_size &&
		x_out >= 0 && x_out < x_size) {
		//printf("(bx,by,bz) = (%d,%d,%d)\n", bx,by,bz);
		//printf("(tx,ty,tz) = (%d,%d,%d)\n", tx,ty,tz);
		//printf("(z_out, y_out, x_out) = (%d,%d,%d)\n", z_out, y_out, x_out);

		float res = 0.0f;
		for (int z_kernel = 0; z_kernel < KERNEL_WIDTH; z_kernel++)
		{
			for (int y_kernel = 0; y_kernel < KERNEL_WIDTH; y_kernel++)
			{
				for (int x_kernel = 0; x_kernel < KERNEL_WIDTH; x_kernel++)
				{
					int z_in = z_out - KERNEL_RADIUS + z_kernel;
					int y_in = y_out - KERNEL_RADIUS + y_kernel;
					int x_in = x_out - KERNEL_RADIUS + x_kernel;
					// Pad boundary with 0
					if (z_in >= 0 && z_in < z_size &&
						y_in >= 0 && y_in < y_size &&
						x_in >= 0 && x_in < x_size) {
						float k_val = M[z_kernel * (KERNEL_WIDTH * KERNEL_WIDTH) + y_kernel * KERNEL_WIDTH + x_kernel];
						float in_val = A[z_in * (y_size * x_size) + y_in * (x_size)+x_in];
						res += k_val * in_val;
					}
				}
			}
		}
		__syncthreads();
		B[z_out * (y_size * x_size) + y_out * (x_size)+x_out] = res;
	}
}

int main(int argc, char* argv[]) {
	wbArg_t args;
	int z_size;
	int y_size;
	int x_size;
	int inputLength, kernelLength;
	float * hostInput;
	float * hostKernel;
	float * hostOutput;
	float * deviceInput;
	float * deviceOutput;

	args = wbArg_read(argc, argv);

	// Import data
	hostInput = (float*)wbImport(wbArg_getInputFile(args, 0), &inputLength);
	hostKernel = (float*)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
	hostOutput = (float*)malloc(inputLength * sizeof(float));

	// First three elements are the input dimensions  
	z_size = hostInput[0];
	y_size = hostInput[1];
	x_size = hostInput[2];
	wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
	assert(z_size * y_size * x_size == inputLength - 3);
	assert(kernelLength == 27);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	//@@ Allocate GPU memory here
	// Recall that inputLength is 3 elements longer than the input data
	// because the first  three elements were the dimensions
	size_t sizeInput = (inputLength - 3) * sizeof(float);
	size_t sizeOutput = sizeInput;
	printf("sizeInput: %d\n", (int)sizeInput);
	printf("sizeOutput: %d\n", (int)sizeOutput);
	wbCheck(cudaMalloc((void **)&deviceInput, sizeInput));
	wbCheck(cudaMalloc((void **)&deviceOutput, sizeOutput));
	size_t sizeKernel = kernelLength * sizeof(float);
	wbTime_stop(GPU, "Doing GPU memory allocation");


	wbTime_start(Copy, "Copying data to the GPU");
	//@@ Copy input and kernel to GPU here
	// Recall that the first three elements of hostInput are dimensions and do
	// not need to be copied to the gpu
	wbCheck(cudaMemcpy(deviceInput, &hostInput[3], sizeInput, cudaMemcpyHostToDevice));
	wbCheck(cudaMemcpyToSymbol(M, hostKernel, sizeKernel));
	wbTime_stop(Copy, "Copying data to the GPU");

	wbTime_start(Compute, "Doing the computation on the GPU");
	//@@ Initialize grid and block dimensions here
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH);
	dim3 dimGrid((x_size - 1) / O_TILE_WIDTH + 1, (y_size - 1) / O_TILE_WIDTH + 1, (z_size - 1) / O_TILE_WIDTH + 1);
	printf("dimBlock = (%d,%d,%d)\n", dimBlock.x, dimBlock.y, dimBlock.z);
	printf("dimGrid = (%d,%d,%d)\n", dimGrid.x, dimGrid.y, dimGrid.z);

	//@@ Launch the GPU kernel here
	conv3d << <dimGrid, dimBlock >> >(deviceInput, deviceOutput, z_size, y_size, x_size);
	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Doing the computation on the GPU");


	wbTime_start(Copy, "Copying data from the GPU");
	//@@ Copy the device memory back to the host here
	// Recall that the first three elements of the output are the dimensions
	// and should not be set here (they are set below)
	wbTime_stop(Copy, "Copying data from the GPU");
	wbCheck(cudaMemcpy(&hostOutput[3], deviceOutput, sizeOutput, cudaMemcpyDeviceToHost)); //the output matrix
	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	// Set the output dimensions for correctness checking
	hostOutput[0] = z_size;
	hostOutput[1] = y_size;
	hostOutput[2] = x_size;
	wbSolution(args, hostOutput, inputLength);

	// Free device memory
	cudaFree(deviceInput);
	cudaFree(deviceOutput);

	// Free host memory
	free(hostInput);
	free(hostOutput);
	return 0;
}

