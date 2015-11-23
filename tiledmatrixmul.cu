#include <wb.h>

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
	    }                                                                          \
    } while (0)

static const float TILE_WIDTH = 8.0;
//static const float SUBTILE_WIDTH = 4.0;

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
	int numAColumns, int numBRows, int numBColumns,
	int numCRows, int numCColumns) {
	//@@ Insert code to implement matrix multiplication here
	//printf("Call to matrixMultiply");
	//int m = numCRows; //equal to numARows
	//int n = numAColumns; //equal to numBRows
	//int k = numCColumns; //equal to numBColumns
	//int Row = blockIdx.y*blockDim.y+threadIdx.y;
	//int Col = blockIdx.x*blockDim.x+threadIdx.x;

	//intf("m: %d, n: %d, k: %d, Row: %d, Col: %d, blockIdx.y: %d, blockDim.y: %d, threadidx.y: %d, blockIdx.x: %d, blockDim.x: %d, threadIdx.x: %d\n", m, n, k, Row, Col, blockIdx.y, blockDim.y, threadIdx.y, blockIdx.x, blockDim.x, threadIdx.x);

	//if((Row < m) && (Col < k)) {
	//		float Cvalue = 0.0;
	//for (int i=0; i<n; ++i)
	//{
	/* A[Row, i] and B[i, Col] */
	//			Cvalue += A[Row*n+i]*B[Col+i*k];
	//}
	//C[Row*k + Col] = Cvalue;
	//printf("C[%d][%d]: %f\n", Row, Col, C[Row*k + Col]);
	//}

	__shared__ float temp_A[(int)TILE_WIDTH][(int)TILE_WIDTH];
	__shared__ float temp_B[(int)TILE_WIDTH][(int)TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int m = numCRows; //equal to numARows
	int n = numAColumns; //equal to numBRows
	int k = numCColumns; //equal to numBColumns

	// Identify the row and column in C
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	float Cvalue = 0;

	//loop through tiles
	for (int tile = 0; tile < ((n - 1) / (TILE_WIDTH)) + 1; ++tile) {

		//without valid row and col check
		//temp_A[ty][tx] = A[Row*n + tile*(int)TILE_WIDTH+tx];
		//temp_B[ty][tx] = B[(tile*(int)TILE_WIDTH+ty)*k + Col];

		//row check
		if (Row < m && tile*(int)TILE_WIDTH + tx < n) {
			temp_A[ty][tx] = A[Row*n + (tile*(int)TILE_WIDTH) + tx];
		}
		//invalid row
		else {
			temp_A[ty][tx] = 0.0;
		}

		//col check
		if (Col < k && tile*(int)TILE_WIDTH + ty < n) {
			temp_B[ty][tx] = B[(tile*(int)TILE_WIDTH + ty)*k + Col];
		}
		//invalid col
		else {
			temp_B[ty][tx] = 0.0;
		}

		//sync threads to make sure all tile elements are loaded
		__syncthreads();

		//the meat
		for (int i = 0; i<TILE_WIDTH; ++i) {
			Cvalue += temp_A[ty][i] * temp_B[i][tx];
		}
		//sync threads to make sure all meat is done
		__syncthreads();

	}

	//write to output matrix
	C[Row*k + Col] = Cvalue;
}


int main(int argc, char **argv) {
	wbArg_t args;
	float *hostA; // The A matrix
	float *hostB; // The B matrix
	float *hostC; // The output C matrix
	float *deviceA;
	float *deviceB;
	float *deviceC;
	int numARows;    // number of rows in the matrix A
	int numAColumns; // number of columns in the matrix A
	int numBRows;    // number of rows in the matrix B
	int numBColumns; // number of columns in the matrix B
	int numCRows;    // number of rows in the matrix C (you have to set this)
	int numCColumns; // number of columns in the matrix C (you have to set this)
	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostA =
		(float *)wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
	hostB =
		(float *)wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
	//@@ Set numCRows and numCColumns
	numCRows = numARows;
	numCColumns = numBColumns;
	//@@ Allocate the hostC matrix
	int sizeC = numCRows * numCColumns * sizeof(float);
	wbLog(TRACE, "C is of size ", sizeC);
	hostC = (float *)malloc(sizeC);
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
	wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
	wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);
	wbTime_start(GPU, "Allocating GPU memory.");
	//@@ Allocate GPU memory here
	int sizeA = numARows * numAColumns * sizeof(float);
	cudaError_t err0 = cudaMalloc((void **)&deviceA, sizeA);
	if (err0 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err0), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	int sizeB = numBRows * numBColumns * sizeof(float);
	cudaError_t err1 = cudaMalloc((void **)&deviceB, sizeB);
	if (err1 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err1), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cudaError_t err8 = cudaMalloc((void **)&deviceC, sizeC);
	if (err8 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err8), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}  wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	//@@ Copy memory to the GPU here
	cudaError_t err2 = cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
	if (err2 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cudaError_t err3 = cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);
	if (err3 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err3), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	//@@ Initialize the grid and block dimensions here
	//dim3 DimGrid = ( ceil(numCColumns/TILE_WIDTH), ceil(numCRows/TILE_WIDTH), 1); //2D grid
	//dim3 DimGrid = ( (numCColumns-1)/TILE_WIDTH + 1, (numCRows-1)/TILE_WIDTH + 1);
	//dim3 DimBlock = ( TILE_WIDTH, TILE_WIDTH); //2D block
	dim3 DimGrid;
	DimGrid.x = (numCColumns - 1) / TILE_WIDTH + 1;
	DimGrid.y = (numCRows - 1) / TILE_WIDTH + 1;
	dim3 DimBlock;
	DimBlock.x = TILE_WIDTH;
	DimBlock.y = TILE_WIDTH;
	//printf("DimGrid: (%d, %d, %d)\n", DimGrid.x, DimGrid.y, DimGrid.z);
	//printf("DimBlock: (%d, %d, %d)\n", DimBlock.x, DimBlock.y, DimBlock.z);

	wbTime_start(Compute, "Performing CUDA computation");
	//@@ Launch the GPU Kernel here
	matrixMultiply << <DimGrid, DimBlock >> >(deviceA, deviceB, deviceC, numARows,
		numAColumns, numBRows, numBColumns,
		numCRows, numCColumns);
	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	//@@ Copy the GPU memory back to the CPU here
	cudaError_t err4 = cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);
	if (err4 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err4), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	//@@ Free the GPU memory here
	cudaError_t err5 = cudaFree(deviceA);
	if (err5 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err5), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cudaError_t err6 = cudaFree(deviceB);
	if (err5 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err6), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cudaError_t err7 = cudaFree(deviceC);
	if (err5 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err7), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, hostC, numCRows, numCColumns);

	free(hostA);
	free(hostB);
	free(hostC);

	return 0;
}
