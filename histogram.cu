// Histogram Equalization

#include <wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
		        }                                                                     \
	    } while(0)

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256
#define SCAN_SIZE (2*BLOCK_SIZE)

//@@ kernel definitions
__global__ void floatToUnsignedChar(float * input, unsigned char * output, int size) {

	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i<size) {
		output[i] = (unsigned char)(255 * input[i]);
		if (i == 0) { printf("floatToUnsignedChar i: %d\t", i); printf("input[i]: %f\t", input[i]); printf("output[i]: %u\n", output[i]); } //debugging
	}
}

__global__ void rgbToGrayscale(unsigned char * input, unsigned char * output, int size) {

	unsigned char r, g, b;
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i<size) {
		r = input[3 * i];
		g = input[3 * i + 1];
		b = input[3 * i + 2];
		output[i] = (unsigned char)(0.21*(float)r + 0.71*(float)g + 0.07*(float)b);
		if (i == 0) { printf("rgbToGrayScale i: %d\t", i); printf("(r,g,b): (%u,%u,%u)\t", r, g, b); printf("output[i]: %u\n", output[i]); } //debugging
	}
}

__global__ void hist(unsigned char * buffer, unsigned int * histo, int size) {
	/*
	histogram = [0, ...., 0] # here len(histogram) = 256
	for ii from 0 to width * height do
	histogram[grayImage[idx]]++
	end
	*/
	__shared__ unsigned int histo_private[HISTOGRAM_LENGTH];

	if (threadIdx.x < HISTOGRAM_LENGTH) histo_private[threadIdx.x] = 0;
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	// stride is the total number of threads in grid (threads/block*blocks/grid)
	int stride = blockDim.x * gridDim.x;

	// All threads handle blockDim.x * gridDim.x consecutive elements
	while (i<size) {
		atomicAdd(&(histo[buffer[i]]), 1);
		i += stride;
	}
	// wait for all other threads in the block to finish
	__syncthreads();

	if (threadIdx.x < HISTOGRAM_LENGTH) {
		atomicAdd(&(histo[threadIdx.x]), histo_private[threadIdx.x]);
	}

	if (threadIdx.x == 0 && blockIdx.x == 0) {
		printf("hist (bx,tx)=(%d,%d): %u\n", blockIdx.x, threadIdx.x, histo[threadIdx.x]);
	}
}

__global__ void calcCDF(float * cdf, unsigned int * histo,
	int imageWidth, int imageHeight, int length) {

	//performs scan on histogram, write to cdf
	/*
	cdf[0] = p(histogram[0])
	for ii from 1 to 256 do
	cdf[ii] = cdf[ii - 1] + p(histogram[ii])
	end
	def p(x):
	return x / (width * height)
	end
	*/

	__shared__ float partialScan[SCAN_SIZE];
	int i = threadIdx.x + blockDim.x*blockIdx.x;

	//load phase, calculate p
	if (i<SCAN_SIZE && i<length) partialScan[i] = (float)histo[i] / (float)(imageWidth * imageHeight);
	if (i == 0) printf("load phase calcCDF[%d]: %f\n", i, partialScan[i]);
	__syncthreads();

	//work efficient reduction phase
	for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
		unsigned int index = (threadIdx.x + 1)*stride * 2 - 1;
		if (index<SCAN_SIZE && index < length)
			partialScan[index] += partialScan[index - stride];
		__syncthreads();
	}

	//work efficient post reduction phase
	for (unsigned int stride = BLOCK_SIZE / 2; stride>0; stride /= 2) {
		__syncthreads();
		unsigned int index = (threadIdx.x + 1)*stride * 2 - 1;
		if (index + stride < SCAN_SIZE && index + stride < length) {
			partialScan[index + stride] += partialScan[index];
		}
	}

	__syncthreads();
	if (i<SCAN_SIZE && i<length)
		cdf[i] += partialScan[threadIdx.x];
	if (i == 0)
		printf("cdf[%d]: %f\n", i, cdf[i]);
}

__global__ void histEqualize(unsigned char * ucharImage, float * cdf, int size) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float cdfmin = cdf[0];
	if (i<size) {
		float x = 255.0F * (cdf[ucharImage[i]] - cdfmin) / (1.0F - cdfmin);
		float start = 0.0F;
		float end = 255.0F;
		if (start > x) x = start;
		if (end < x) x = end;
		ucharImage[i] = (unsigned char)x;
	}
	if (i == 0) {
		printf("histEqualize[%d]: %u\n", i, ucharImage[i]);
	}
}

__global__ void unsignedCharToFloat(unsigned char * input, float * output, int size) {

	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i<size) {
		output[i] = (float)(input[i] / 255.0);
		if (i == 0) { printf("unsignedCharToFloat i: %d\t", i); printf("input[i]: %u\t", input[i]); printf("output[i]: %f\n", output[i]); } //debugging
	}
}


int main(int argc, char **argv) {
	wbArg_t args;
	int imageWidth;
	int imageHeight;
	int imageChannels;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	const char *inputImageFile;

	float *deviceInputImageData;
	float *deviceOutputImageData;
	unsigned int * histo;
	unsigned char * uCharImage;
	unsigned char * grayImage;
	float * cdf;

	//@@ Insert more code here

	args = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(args, 0);

	wbTime_start(Generic, "Importing data and creating memory on host");
	inputImage = wbImport(inputImageFile);
	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);
	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

	wbLog(TRACE, "Input image width: ", imageWidth);
	wbLog(TRACE, "Input image height: ", imageHeight);
	wbLog(TRACE, "Input image channels: ", imageChannels);

	hostInputImageData = wbImage_getData(inputImage); //Piazza 204
	hostOutputImageData = wbImage_getData(outputImage); //Piazza 204
	wbTime_stop(Generic, "Importing data and creating memory on host");

	//@@ insert code here

	//@@ Allocate GPU memory here
	wbTime_start(GPU, "Allocating GPU memory.");
	wbCheck(cudaMalloc((void **)&deviceInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float)));
	wbCheck(cudaMalloc((void **)&uCharImage, imageWidth*imageHeight*imageChannels*sizeof(unsigned char)));
	wbCheck(cudaMalloc((void **)&grayImage, imageWidth*imageHeight*sizeof(unsigned int)));
	wbCheck(cudaMalloc((void **)&histo, HISTOGRAM_LENGTH*sizeof(unsigned int)));
	wbCheck(cudaMalloc((void **)&cdf, HISTOGRAM_LENGTH*sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceOutputImageData, imageWidth*imageHeight*imageChannels*sizeof(float)));
	wbTime_stop(GPU, "Allocating GPU memory.");

	//@@ Copy memory to the GPU here
	wbTime_start(Copy, "Copying input image to GPU.");
	wbCheck(cudaMemcpy(deviceInputImageData,
		hostInputImageData,
		imageWidth*imageHeight*imageChannels*sizeof(float),
		cudaMemcpyHostToDevice));
	wbTime_stop(Copy, "Copying input image to GPU.");
	wbTime_start(Copy, "Zeroing out histogram.");
	wbCheck(cudaMemset(histo, 0, HISTOGRAM_LENGTH*sizeof(unsigned int)));
	wbTime_stop(Copy, "Zeroing out histogram.");
	wbTime_start(Copy, "Zeroing out cdf.");
	wbCheck(cudaMemset(cdf, 0, HISTOGRAM_LENGTH*sizeof(float)));
	wbTime_stop(Copy, "Zeroing out cdf.");

	//@@ Initialize the grid and block dimensions here
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(((imageWidth*imageHeight*imageChannels) - 1) / BLOCK_SIZE + 1);
	dim3 dimGridHist(((imageWidth*imageHeight*imageChannels) - 1) / HISTOGRAM_LENGTH + 1);

	//@@ Launch the GPU kernel to cast the image from float to unsigned char
	wbTime_start(Compute, "Performing CUDA mask to uCharImage.");
	floatToUnsignedChar << <dimGrid, dimBlock >> >(deviceInputImageData, uCharImage, imageWidth*imageHeight*imageChannels);
	wbTime_stop(Compute, "Performing CUDA mask to uCharImage.");

	//@@ Launch the GPU kernel to convert the image from RGB to GrayScale
	wbTime_start(Compute, "Performing CUDA converstion to grayImage.");
	rgbToGrayscale << <dimGrid, dimBlock >> >(uCharImage, grayImage, imageWidth*imageHeight);
	wbTime_stop(Compute, "Performing CUDA converstion to grayImage.");

	//@@ Launch the GPU kernel to compute the Cumulative Distribution Function of histogram
	wbTime_start(Compute, "Performing CUDA CDF histogram.");
	hist << <dimGrid, dimBlock >> >(grayImage, histo, imageWidth*imageHeight);
	wbTime_stop(Compute, "Performing CUDA CDF histogram.");

	//@@ Launch the GPU kernel to compute the CDF function of histogram
	wbTime_start(Compute, "Calculating CDF.");
	calcCDF << <dimGridHist, dimBlock >> >(cdf, histo, imageWidth, imageHeight, HISTOGRAM_LENGTH);
	wbTime_stop(Compute, "Calculating CDF.");

	//@@ Calculate mincdf and initialize cdf[0] (see piazza 201)
	wbTime_start(Compute, "Calculating mincdf.");
	wbTime_stop(Compute, "Calculating mincdf.");

	//@@ Launch the GPU kernel to apply the histogram equalization function
	wbTime_start(Compute, "Performing CUDA histogram equalization function.");
	histEqualize << <dimGrid, dimBlock >> >(uCharImage, cdf, imageWidth*imageHeight*imageChannels);
	wbTime_stop(Compute, "Performing CUDA histogram equalization function.");

	//@@ Launch the GPU kernel to cast the image from float to unsigned char
	wbTime_start(Compute, "Performing CUDA from float to unsigned char.");
	unsignedCharToFloat << <dimGrid, dimBlock >> >(uCharImage, deviceOutputImageData, imageWidth*imageHeight*imageChannels);
	wbTime_stop(Compute, "Performing CUDA from float to unsigned char.");

	//@@ Copy the GPU memory back to the CPU here
	wbCheck(cudaMemcpy(hostOutputImageData,
		deviceOutputImageData,
		imageWidth*imageHeight*imageChannels*sizeof(float),
		cudaMemcpyDeviceToHost));

	//@@ Output image format
	wbImage_setData(outputImage, hostOutputImageData); //Piazza 204

	//@@ Free the GPU memory here
	wbCheck(cudaFree(deviceInputImageData));
	wbCheck(cudaFree(uCharImage));
	wbCheck(cudaFree(grayImage));
	wbCheck(cudaFree(histo));
	wbCheck(cudaFree(cdf));
	wbCheck(cudaFree(deviceOutputImageData));


	//@@ Check Solution
	wbSolution(args, outputImage);

	//@@ insert code here

	return 0;
}
