// MP 1
#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
	//@@ Insert code to implement vector addition here
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i<len)
	{
		//printf("in1[%d]: %d\n", i, in1[i]);
		//printf("in2[%d]: %d\n", i, in2[i]);
		out[i] = in1[i] + in2[i];
		//printf("out[%d]: %d\n", i, out[i]);
	}
}


int main(int argc, char **argv) {
	wbArg_t args;
	int inputLength;
	float *hostInput1;
	float *hostInput2;
	float *hostOutput;
	float *deviceInput1;
	float *deviceInput2;
	float *deviceOutput;

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
	hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
	hostOutput = (float *)malloc(inputLength * sizeof(float));
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The input length is ", inputLength);

	wbTime_start(GPU, "Allocating GPU memory.");
	//@@ Allocate GPU memory here

	int size = inputLength * sizeof(float);
	wbLog(TRACE, "size is ", size);
	cudaError_t err0 = cudaMalloc((void **)&deviceInput1, size); //address of pointer, size
	if (err0 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err0), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cudaError_t err1 = cudaMalloc((void **)&deviceInput2, size);
	if (err1 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err1), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	cudaError_t err2 = cudaMalloc((void **)&deviceOutput, size);
	if (err2 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	//@@ Copy memory to the GPU here
	cudaError_t err3 = cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice); //dest, src, size, cudaMemcpyHostToDevice
	if (err3 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err3), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cudaError_t err4 = cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);
	if (err4 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err4), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	//@@ Initialize the grid and block dimensions here
	//dim3 DimGrid(ceil(size/256.0),1,1);
	//dim3 DimBlock(256,1,1);
	int numBlocks = ceil(size / 256.0);
	int threadsPerBlock = 256;
	wbLog(TRACE, "There are ", numBlocks, " blocks");
	wbLog(TRACE, "There are ", threadsPerBlock, " threads per block");

	wbTime_start(Compute, "Performing CUDA computation");
	//@@ Launch the GPU Kernel here
	//vecAdd<<<DimGrid,DimBlock>>>(deviceInput1, deviceInput2, deviceOutput, size);
	vecAdd << <numBlocks, threadsPerBlock >> >(deviceInput1, deviceInput2, deviceOutput, size);

	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	//@@ Copy the GPU memory back to the CPU here
	cudaError_t err5 = cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost); //dest, src, size, cudaMemcpyDeviceToHost
	if (err5 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err5), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	//@@ Free the GPU memory here
	cudaError_t err6 = cudaFree(deviceInput1);
	if (err6 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err6), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cudaError_t err7 = cudaFree(deviceInput2);
	if (err7 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err7), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cudaError_t err8 = cudaFree(deviceOutput);
	if (err8 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err8), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, hostOutput, inputLength);

	free(hostInput1);
	free(hostInput2);
	free(hostOutput);

	return 0;
}
