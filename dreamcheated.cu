#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

const int PEARLTHRESHOLD = 20;
const int PEARLRANGE = 423;
const int GOLDTRADES = 262;
const int DREAMPEARLS = 42;

const int BLAZETHRESHOLD = 1;
const int BLAZERANGE = 2;
const int BLAZEKILLS = 305;
const int DREAMRODS = 211;

const int LOGFREQ = 1000000;


static const int wholeArraySize = 100000000;
static const int blockSize = 1024;
static const int gridSize = 24;


  int N = 1<<20; // ~1 million

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

__global__ void setup_kernel(curandState * state, long int seed){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n; i += stride) {
    curand_init(seed, index, 0, &state[index]);
  }
};



__device__ bool lastBlock(int* counter) {
    __threadfence(); //ensure that partial result is visible by all blocks
    int last = 0;
    if (threadIdx.x == 0)
        last = atomicAdd(counter, 1);
    return __syncthreads_or(last == gridDim.x-1);
}

__global__ void sumCommMultiBlock(const int *gArr, int arraySize, int *gOut, int* lastBlockCounter) {
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x*blockSize;
    const int gridSize = blockSize*gridDim.x;
    int sum = 0;
    for (int i = gthIdx; i < arraySize; i += gridSize)
        sum += gArr[i];
    __shared__ int shArr[blockSize];
    shArr[thIdx] = sum;
    __syncthreads();
    for (int size = blockSize/2; size>0; size/=2) { //uniform
        if (thIdx<size)
            shArr[thIdx] += shArr[thIdx+size];
        __syncthreads();
    }
    if (thIdx == 0)
        gOut[blockIdx.x] = shArr[0];
    if (lastBlock(lastBlockCounter)) {
        shArr[thIdx] = thIdx<gridSize ? gOut[thIdx] : 0;
        __syncthreads();
        for (int size = blockSize/2; size>0; size/=2) { //uniform
            if (thIdx<size)
                shArr[thIdx] += shArr[thIdx+size];
            __syncthreads();
        }
        if (thIdx == 0)
            gOut[0] = shArr[0];
    }
}


__global__ void rodsAndPearls(curandState * my_curandstate, int *rods, int *pearls){
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  // rod section
  int rodCount = 0;
  for (int j = 0; j < BLAZEKILLS; j++){
    int drop;
    float r = curand_uniform(&(my_curandstate[index]));

    drop = ceil(r * 2) - 1
    rodCount += drop;

  }
  rods[index] = rodCount;

  // pearl section
  int pearlCount = 0;
  for (int j = 0; j < GOLDTRADES; j++){
    int trade;

    float r = curand_uniform(&(my_curandstate[index]));
    if (r <= PEARLTHRESHOLD) {
      pearlCount++;
    }
  }

  pearls[index] = pearlCount;

  // sum section



}

int main() {
  FILE * fp;
  if ((fp = fopen("dream.txt", "r")) == NULL){
      printf("File not found");
      exit(1);
  }

  // open file
  FILE *fp;
  if ((fp = fopen("dream.txt", "r")) == NULL){
      printf("File not found");
      exit(1);
  }

  // find relevant data
  fscanf(fp, "%i,%i,%llu,%19[^\n],%llu,%lf,%lf", &maxRods, &maxPearls, &maxAttempts, whenFound, &attempts, &totalExecTime, &speed);

  // close file
  fclose(fp);

  printf("%i\n", maxRods);
  printf("%i\n", maxPearls);
  printf("%llu\n", maxAttempts);
  printf("%s\n", whenFound);
  printf("%llu\n", attempts);
  printf("%lf\n", totalExecTime);

  long seed;

  float *rods, *pearls;

  cudaMalloc(&rods, N*sizeof(float));
  cudaMalloc(&pearls, N*sizeof(float));

  int i = 0;

  while (i < 1){



    long int randSeed;
    getrandom(&randSeed, sizeof(randSeed), 0);
    long int seed = (long int)time(NULL) ^= randSeed;

    curandState *d_state;
    cudaMalloc(&d_state, sizeof(curandState));
    setup_kernel<<<numBlocks, blockSize>>>(d_state, seed);


    rodsAndPearls<<<numBlocks, blockSize>>>(d_state, rods, pearls)

    int * maxRodBatch;
    int * maxPearlBatch;

    sumRodsPearls<<<numBlocks, blockSize>>>(d_state, rods, pearls, maxRodBlock,
      maxPearlBatch);




    i++;
  }

}
