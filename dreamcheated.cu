#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <sys/random.h>

const int PEARLTHRESHOLD = 20;
const int PEARLRANGE = 423;
const int GOLDTRADES = 262;
const int DREAMPEARLS = 42;

const int BLAZETHRESHOLD = 1;
const int BLAZERANGE = 2;
const int BLAZEKILLS = 305;
const int DREAMRODS = 211;

const int LOGFREQ = 1000000;


//static const int wholeArraySize = 100000000;
//static const int blockSize = 1024;
//static const int gridSize = 24;


int N = 1<<20; // ~1 million

int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;


/* this GPU kernel function is used to initialize the random states */
__global__ void setup_kernel(unsigned int seed, curandState_t* states) {

  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[blockIdx.x]);
}



__global__ void maxReduce(int *g_idata, int *g_odata) {
  extern __shared__ int sdata[];

  // each thread loads one element from global to shared memory
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tid = threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();

  // do reduction
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) {
      sdata[tid] = max(sdata[tid+s], sdata[tid]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}


__global__ void rodsAndPearls(curandState_t * my_curandstate,
    unsigned long long int * rods, unsigned long long int * pearls,
    unsigned int maxRods, unsigned int maxPearls) {

  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  // rod section
  unsigned int rodCount = 0;


  for (int i = 0; i < 9; i++) { // sum first 288 bits
      rodCount += __popc(curand(&my_curandstate[index]) % UINT_MAX);
  }

  unsigned int extra = (curand(&my_curandstate[index]) % UINT_MAX) & (1<<17 - 1);
  rodCount += __popc(extra); // add last 17 bits

  if (rodCount > maxRods) {
    ;
  } else {
    rodCount = 0;
  }

}




int main() {

  // open file
  FILE *fp;
  if ((fp = fopen("dream.txt", "r")) == NULL){
      printf("File not found");
      exit(1);
  }

  int maxRods;
  int maxPearls;
  unsigned long long maxAttempts;
  unsigned long long attempts;
  char whenFound[19];
  double totalExecTime;
  double speed;

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

  unsigned long long int * rods;
  unsigned long long int * pearls;

  cudaMalloc(&rods, N*sizeof(uint64_t));
  cudaMalloc(&pearls, N*sizeof(uint64_t));

  int i = 0;

  while (i < 1) {

    long int seed = (long int)time(NULL);

    curandState_t *d_state;
    cudaMalloc(&d_state, sizeof(curandState_t));
    setup_kernel<<<numBlocks, blockSize>>>(seed, d_state);

    printf("burh");

    rodsAndPearls<<<numBlocks, blockSize>>>(d_state, rods, pearls, maxRods, maxPearls);

    for (int j = 0; j < 256; j++) {
      printf("%llu, ", rods[j]);
    }


    i++;
  }
}
