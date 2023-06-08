# PCA-Demonstrate-Matrix-transposition-on-shared-memory
Comparing the Performance of the Rectangular Shared Memory Kernels with  grid (1,1) block (16,16)

## AIM:

To Demonstrate Matrix transposition on shared memory and Compare the Performance of the
Rectangular Shared Memory Kernels with grid (1,1) block (16,16).

## PROCEDURE:

1. Allocate memory on the GPU for the input matrix and output matrix.

2. Copy the input matrix from the host to the GPU memory.

3. Define the kernel function for matrix transposition using shared memory.

4. Allocate shared memory on the GPU for the input and output matrices.

5. Load a tile of the input matrix into shared memory.

6. Use synchronization to ensure all threads have finished loading the tile into shared memory.

7. Transpose the tile in shared memory.

8.  Use synchronization to ensure all threads have finished transposing the tile.

9.  Write the transposed tile back to global memory.

10. Repeat steps 5-9 until the entire input matrix has been transposed.

11. Copy the transposed matrix from the GPU memory to the host.

12. Measure the time taken for the matrix transposition using rectangular shared memory
kernels and grid (1,1) block (16,16).

13. Compare the performance of the two methods.

## PROGRAM:
### checkSmemRectangle.cu:
```
#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#define BDIMX 16
#define BDIMY 16
#define IPAD 2
void printData(char *msg, int *in, const int size)
{
printf("%s: ", msg);
for (int i = 0; i < size; i++)
{
printf("%4d",
in[i]);
fflush(stdout);
}
printf("\n\n");
}
 global void setRowReadRow(int *out)
{
// static shared memory
 shared int tile[BDIMY][BDIMX];
// mapping from thread index to global memory index
unsigned int idx = threadIdx.y * blockDim.x +
threadIdx.x;
// shared memory store operation
tile[threadIdx.y][threadIdx.x] =
idx;
// wait for all threads to complete
 syncthreads();
// shared memory load operation
out[idx] = tile[threadIdx.y][threadIdx.x] ;
}
 global void setColReadCol(int *out)
{
// static shared memory
 shared int tile[BDIMX][BDIMY];
// mapping from thread index to global memory index
unsigned int idx = threadIdx.y * blockDim.x +
threadIdx.x;
// shared memory store operation
tile[threadIdx.x][threadIdx.y] = idx;
// wait for all threads to complete
 syncthreads();
// shared memory load operation
out[idx] = tile[threadIdx.x][threadIdx.y];
}
 global void setColReadCol2(int *out)
{
// static shared memory
 shared int tile[BDIMY][BDIMX];
// mapping from 2D thread index to linear memory
unsigned int idx = threadIdx.y * blockDim.x +
threadIdx.x;
// convert idx to transposed coordinate (row, col)unsigned
int irow = idx / blockDim.y;
unsigned int icol = idx % blockDim.y;
// shared memory store operation
tile[icol][irow] = idx;
// wait for all threads to complete
 syncthreads();
// shared memory load operation
out[idx] = tile[icol][irow] ;
}
 global void setRowReadCol(int *out)
{
// static shared memory
 shared int tile[BDIMY][BDIMX];
// mapping from 2D thread index to linear memory
unsigned int idx = threadIdx.y * blockDim.x +
threadIdx.x;
// convert idx to transposed coordinate (row, col)unsigned
int irow = idx / blockDim.y;
unsigned int icol = idx % blockDim.y;
// shared memory store operation
tile[threadIdx.y][threadIdx.x] =
idx;
// wait for all threads to complete
 syncthreads();
// shared memory load operation
out[idx] = tile[icol][irow];
}
 global void setRowReadColPad(int *out)
{
// static shared memory
 shared int tile[BDIMY][BDIMX + IPAD];
// mapping from 2D thread index to linear memory
unsigned int idx = threadIdx.y * blockDim.x +
threadIdx.x;
// convert idx to transposed (row, col)
unsigned int irow = idx / blockDim.y;
unsigned int icol = idx % blockDim.y;
// shared memory store operation
tile[threadIdx.y][threadIdx.x] =
idx;
// wait for all threads to complete
 syncthreads();
// shared memory load operation
out[idx] = tile[icol][irow] ;
}
 global void setRowReadColDyn(int *out)
{
// dynamic shared memory
extern shared int tile[];
// mapping from thread index to global memory index
unsigned int idx = threadIdx.y * blockDim.x +
threadIdx.x;
// convert idx to transposed (row, col)
unsigned int irow = idx / blockDim.y;
unsigned int icol = idx % blockDim.y;
// convert back to smem idx to access the transposed element
unsigned int col_idx = icol * blockDim.x + irow;
// shared memory store operation
tile[idx] = idx;
// wait for all threads to complete
 syncthreads();
// shared memory load operation
out[idx] = tile[col_idx];
}
 global void setRowReadColDynPad(int *out)
{
// dynamic shared memory
extern shared int tile[];
// mapping from thread index to global memory index unsigned
int g_idx = threadIdx.y * blockDim.x + threadIdx.x;
// convert idx to transposed (row, col)
unsigned int irow = g_idx /
blockDim.y;unsigned int icol = g_idx
% blockDim.y;
unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
// convert back to smem idx to access the transposed element
unsigned int col_idx = icol * (blockDim.x + IPAD) + irow;
// shared memory store operation
tile[row_idx] = g_idx;
// wait for all threads to complete
 syncthreads();
// shared memory load operation
out[g_idx] = tile[col_idx];
}
int main(int argc, char **argv)
{
// set up device
int dev = 0;
cudaDeviceProp deviceProp;
CHECK(cudaGetDeviceProperties(&deviceProp, dev));
printf("%s at ", argv[0]);
printf("device %d: %s ", dev, deviceProp.name);
CHECK(cudaSetDevice(dev));
cudaSharedMemConfig pConfig;
CHECK(cudaDeviceGetSharedMemConfig ( &pConfig ));
printf("with Bank Mode:%s ", pConfig == 1 ? "4-Byte" : "8-Byte");
// set up array size
int nx = BDIMX;
int ny =
BDIMY;bool
iprintf = 0;
if (argc > 1) iprintf = atoi(argv[1]);
size_t nBytes = nx * ny * sizeof(int);
// execution configuration
dim3 block (BDIMX,
BDIMY);dim3 grid (1, 1);
printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);
// allocate device memoryint
*d_C;
CHECK(cudaMalloc((int**)&d_C,
nBytes));int *gpuRef = (int
*)malloc(nBytes);
CHECK(cudaMemset(d_C, 0, nBytes));
setRowReadRow<<<grid, block>>>(d_C);
CHECK(cudaMemcpy(gpuRef, d_C, nBytes,
cudaMemcpyDeviceToHost));if(iprintf) printData("setRowReadRow ",
gpuRef, nx * ny); CHECK(cudaMemset(d_C, 0, nBytes));
setColReadCol<<<grid, block>>>(d_C);
CHECK(cudaMemcpy(gpuRef, d_C, nBytes,
cudaMemcpyDeviceToHost));if(iprintf) printData("setColReadCol ",
gpuRef, nx * ny); CHECK(cudaMemset(d_C, 0, nBytes));
setColReadCol2<<<grid, block>>>(d_C);
CHECK(cudaMemcpy(gpuRef, d_C, nBytes,
cudaMemcpyDeviceToHost));if(iprintf) printData("setColReadCol2 ",
gpuRef, nx * ny); CHECK(cudaMemset(d_C, 0, nBytes));
setRowReadCol<<<grid, block>>>(d_C);
CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
if(iprintf) printData("setRowReadCol ", gpuRef, nx *
ny);CHECK(cudaMemset(d_C, 0, nBytes));
setRowReadColDyn<<<grid, block,
BDIMX*BDIMY*sizeof(int)>>>(d_C); CHECK(cudaMemcpy(gpuRef,
d_C, nBytes, cudaMemcpyDeviceToHost));if(iprintf)
printData("setRowReadColDyn ", gpuRef, nx * ny);
CHECK(cudaMemset(d_C, 0, nBytes));
setRowReadColPad<<<grid, block>>>(d_C);
CHECK(cudaMemcpy(gpuRef, d_C, nBytes,
cudaMemcpyDeviceToHost));if(iprintf) printData("setRowReadColPad
", gpuRef, nx * ny); CHECK(cudaMemset(d_C, 0, nBytes));
setRowReadColDynPad<<<grid, block, (BDIMX + IPAD)*BDIMY*sizeof(int)>>>(d_C);
CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
if(iprintf) printData("setRowReadColDynPad ", gpuRef, nx * ny);
// free host and device memory
CHECK(cudaFree(d_C));
free(gpuRef);
// reset device
CHECK(cudaDeviceReset()
);return EXIT_SUCCESS;
}
```

## OUTPUT:

root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_6# ./checkSmemRectangle
./checkSmemRectangle at device 0: NVIDIA GeForce GT 710 with Bank Mode:4-Byte <<< grid (1,1)
block (16,16)>>>

## RESULT:

Thus, the Matrix transposition on shared memory and Comparing the Performance of the
Rectangular Shared Memory Kernels with grid (1,1) block (16,16) has been successfully performed
