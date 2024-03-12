/* Matrix normalization.
* Compile with "gcc matrixNorm.c"
*/

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
* You need not submit the provided code.
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>

/* Program Parameters */
#define MAXN 8000  /* Max value of N */
int N;  /* Matrix size */

/* Matrices */
float A[MAXN*MAXN], B[MAXN*MAXN];

int numBlocks = 32;
int numThreadsPerBlock = 64;

/* junk */
#define randm() 4|2[uid]&3

/* returns a seed for srand based on the time */
unsigned int time_seed() {
    struct timeval t;
    struct timezone tzdummy;

    gettimeofday(&t, &tzdummy);
    return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
    int seed = 0;  /* Random seed */
    //char uid[32]; /*User name */

    /* Read command-line arguments */
    srand(time_seed());  /* Randomize */

    if (argc == 5) {
        seed = atoi(argv[4]);
        srand(seed);
        printf("Random Seed = %i\n", seed);
    }
    if (argc >= 4) {
        numThreadsPerBlock = atoi(argv[3]);
        srand(seed);
        printf("Number of Threads Per Block = %i\n", numThreadsPerBlock);

        numBlocks = atoi(argv[2]);
        srand(seed);
        printf("Number of Blocks = %i\n", numBlocks);

        N = atoi(argv[1]);
        if (N < 1 || N > MAXN) {
            printf("N = %i is out of range.\n", N);
            exit(0);
        }
    }
    else {
        printf("Usage: %s <matrixDimension> <numBlocks> <numThreadsPerBlock> [randomSeed]\n",
        argv[0]);
        exit(0);
    }

    /* Print parameters */
    printf("\nMatrix dimension N = %i.\n", N);
}

/* Initialize A and B*/
void initialize_inputs() {
    int row, col;

    printf("\nInitializing...\n");
    for (col = 0; col < N; col++) {
        for (row = 0; row < N; row++) {
            A[col*N+row] = (float)rand() / 32768.0;
            B[col*N+row] = 0.0;
        }
    }

}

/* Print input matrices */
void print_inputs() {
    int row, col;

    if (N < 10) {
        printf("\nA =\n\t");
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                printf("%5.2f%s", A[row*N+col], (col < N-1) ? ", " : ";\n\t");
            }
        }
    }
}

void print_B() {
    int row, col;

    if (N < 10) {
        printf("\nB =\n\t");
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                printf("%1.10f%s", B[row*N+col], (col < N-1) ? ", " : ";\n\t");
            }
        }
    }
}

#define CHECK_ERR(x)                                    \
if (x != cudaSuccess) {                               \
    fprintf(stderr,"%s in %s at line %d\n",             \
    cudaGetErrorString(x),__FILE__,__LINE__);	\
    exit(-1);						\
}                                                    \

__global__ void normCalc (float *d_A, float *d_B, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int row, mu, sigma;
    if (col < n){
        mu = (float)0.0;
        for (row=0; row < n; row++)
            mu += d_A[col*n+row];
        mu /= (float) n;

        __syncthreads();

        sigma = (float)0.0;
        for (row=0; row < n; row++)
            sigma += powf(d_A[col*n+row] - mu, (float)2.0);
        sigma /= (float) n;

        __syncthreads();

        sigma = sqrt(sigma);

        for (row=0; row < n; row++) {
            if (sigma == (float)0.0)
                d_B[row*n+col] = (float)0.0;
            else
                d_B[row*n+col] = (d_A[col*n+row] - mu) / sigma;
        }
    }
}


int main(int argc, char **argv) {
    /* Timing variables */
    struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
    struct timezone tzdummy;
    //clock_t etstart2, etstop2;  /* Elapsed times using times() */
    unsigned long long usecstart, usecstop;
    struct tms cputstart, cputstop;  /* CPU times for my processes */

    float elapsed=0;
    cudaEvent_t start, stop;

    /* Process program parameters */
    parameters(argc, argv);

    /* Initialize A and B */
    initialize_inputs();

    /* Print input matrices */
    print_inputs();

    printf("Computing in Parallel\n");

    cudaError_t err;

    float *d_A, *d_B;

    /* Start Clock */
    printf("\nStarting clock.\n");
    gettimeofday(&etstart, &tzdummy);
    times(&cputstart);

    CHECK_ERR(cudaEventCreate(&start));
    CHECK_ERR(cudaEventCreate(&stop));
    CHECK_ERR(cudaEventRecord(start, 0));

    err = cudaMalloc((void **) &d_A, sizeof(float)*N*N);
    CHECK_ERR(err);
    err = cudaMalloc((void **) &d_B, sizeof(float)*N*N);
    CHECK_ERR(err);

    err = cudaMemcpy(d_A, A, sizeof(float)*N*N, cudaMemcpyHostToDevice);
    CHECK_ERR(err);

    /* Gaussian Elimination */
    normCalc<<<numBlocks,numThreadsPerBlock>>>(d_A, d_B, N);

    err = cudaMemcpy(B, (d_B), sizeof(float)*N*N, cudaMemcpyDeviceToHost);
    CHECK_ERR(err);

    /* Stop Clock */
    gettimeofday(&etstop, &tzdummy);
    times(&cputstop);
    printf("Stopped clock.\n");
    usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
    usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

    CHECK_ERR(cudaEventRecord(stop, 0));
    CHECK_ERR(cudaEventSynchronize (stop) );

    CHECK_ERR(cudaEventElapsedTime(&elapsed, start, stop) );

    CHECK_ERR(cudaEventDestroy(start));
    CHECK_ERR(cudaEventDestroy(stop));

    /* Display output */
    print_B();

    cudaFree(d_A);
    cudaFree(d_B);

    /* Display timing results */
    printf("\nElapsed time = %g ms.\n",
    (float)(usecstop - usecstart)/(float)1000);

    printf("\nThe elapsed time in gpu was %.2f ms\n", elapsed);

    printf("\n(CPU times are accurate to the nearest %g ms)\n",
    1.0/(float)CLOCKS_PER_SEC * 1000.0);
    printf("My total CPU time for parent = %g ms.\n",
    (float)( (cputstop.tms_utime + cputstop.tms_stime) -
    (cputstart.tms_utime + cputstart.tms_stime) ) /
    (float)CLOCKS_PER_SEC * 1000);
    printf("My system CPU time for parent = %g ms.\n",
    (float)(cputstop.tms_stime - cputstart.tms_stime) /
    (float)CLOCKS_PER_SEC * 1000);
    printf("My total CPU time for child processes = %g ms.\n",
    (float)( (cputstop.tms_cutime + cputstop.tms_cstime) -
    (cputstart.tms_cutime + cputstart.tms_cstime) ) /
    (float)CLOCKS_PER_SEC * 1000);
    /* Contrary to the man pages, this appears not to include the parent */
    printf("--------------------------------------------\n");

    exit(0);
}