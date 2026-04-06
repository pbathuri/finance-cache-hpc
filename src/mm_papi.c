#include <assert.h>
#include <papi.h>
#include <stdlib.h>

__attribute__((noipa))

void do_not_optimize(float *_) {}

#if defined(ROW_MAJOR)
  #define A_IDX(i,k,M,K) ((i)*(K) + (k))
  #define B_IDX(k,j,K,N) ((k)*(N) + (j))
  #define C_IDX(i,j,M,N) ((i)*(N) + (j))
#elif defined(COL_MAJOR)
  #define A_IDX(i,k,M,K) ((i) + (k)*(M))
  #define B_IDX(k,j,K,N) ((k) + (j)*(K))
  #define C_IDX(i,j,M,N) ((i) + (j)*(M))
#else
  #error "Define ROW_MAJOR or COL_MAJOR"
#endif

int main(int argc, char** argv)
{
  assert(argc == 4);

  PAPI_library_init(PAPI_VER_CURRENT);

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);

  float *A = (float*)calloc((size_t)M * (size_t)K, sizeof(float));
  float *B = (float*)calloc((size_t)K * (size_t)N, sizeof(float));
  float *C = (float*)calloc((size_t)M * (size_t)N, sizeof(float));

  PAPI_hl_region_begin("initialize");

  for (int i = 0; i < M * K; ++i) {
 	A[i] = 1.f / (float)((rand() % 1024) + 1);
}
  for (int i = 0; i < K * N; ++i) {
	B[i] = 1.f / (float)((rand() % 1024) + 1);
}

  PAPI_hl_region_end("initialize");

  PAPI_hl_region_begin("multiply");

#if defined(ORDER_IJK)
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < K; ++k)
        C[C_IDX(i,j,M,N)] += A[A_IDX(i,k,M,K)] * B[B_IDX(k,j,K,N)];

#elif defined(ORDER_IKJ)
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k)
      for (int j = 0; j < N; ++j)
        C[C_IDX(i,j,M,N)] += A[A_IDX(i,k,M,K)] * B[B_IDX(k,j,K,N)];

#elif defined(ORDER_JIK)
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < M; ++i)
      for (int k = 0; k < K; ++k)
        C[C_IDX(i,j,M,N)] += A[A_IDX(i,k,M,K)] * B[B_IDX(k,j,K,N)];

#elif defined(ORDER_JKI)
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < K; ++k)
      for (int i = 0; i < M; ++i)
        C[C_IDX(i,j,M,N)] += A[A_IDX(i,k,M,K)] * B[B_IDX(k,j,K,N)];

#elif defined(ORDER_KIJ)
  for (int k = 0; k < K; ++k)
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j)
        C[C_IDX(i,j,M,N)] += A[A_IDX(i,k,M,K)] * B[B_IDX(k,j,K,N)];

#elif defined(ORDER_KJI)
  for (int k = 0; k < K; ++k)
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < M; ++i)
        C[C_IDX(i,j,M,N)] += A[A_IDX(i,k,M,K)] * B[B_IDX(k,j,K,N)];
#else
  #error "Define one: ORDER_IJK ORDER_IKJ ORDER_JIK ORDER_JKI ORDER_KIJ ORDER_KJI"
#endif

  PAPI_hl_region_end("multiply");

  do_not_optimize(C);

  free(C);
  free(B);
  free(A);

  return 0;
}
