/*
 * cholesky_papi.c
 *
 * PAPI-instrumented Cholesky decomposition of a synthetic SPD matrix.
 *
 * Compile-time selection:
 *   Layout:    -DROW_MAJOR  or  -DCOL_MAJOR
 *   Algorithm: -DALGO_BANACHIEWICZ (row-by-row)
 *              -DALGO_CROUT        (column-by-column)
 *
 * Usage:  ./cholesky_<layout>_<algo>  <d>
 *
 * Prints one CSV line to stdout:
 *   layout,algo,d,gflops,l1_miss,seconds
 *
 * Uses PAPI low-level eventset API (PAPI_create_eventset/PAPI_start/PAPI_stop)
 * for direct in-memory L1 miss reads — no JSON file parsing needed.
 */

#include <assert.h>
#include <math.h>
#include <papi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* ---- layout ---------------------------------------------------- */
#if defined(ROW_MAJOR)
  #define IDX(i,j,n) ((i)*(n) + (j))
  #define LAYOUT_STR "rm"
#elif defined(COL_MAJOR)
  #define IDX(i,j,n) ((i) + (j)*(n))
  #define LAYOUT_STR "cm"
#else
  #error "Define ROW_MAJOR or COL_MAJOR"
#endif

#if defined(ALGO_BANACHIEWICZ)
  #define ALGO_STR "banachiewicz"
#elif defined(ALGO_CROUT)
  #define ALGO_STR "crout"
#else
  #error "Define ALGO_BANACHIEWICZ or ALGO_CROUT"
#endif

void __attribute__((noinline)) do_not_optimize(float *_) { (void)_; }
static volatile float sink_f;

/* ---- build random SPD matrix: A = M M^T + d·I ------------------- */
static void build_spd(float *A, int d)
{
    float *M = (float *)calloc((size_t)d * d, sizeof(float));
    assert(M);
    srand(42);
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            M[IDX(i, j, d)] = (float)(rand() % 1024 + 1) / 1024.0f;
    for (int i = 0; i < d; i++)
        for (int j = 0; j <= i; j++) {
            float s = 0.0f;
            for (int k = 0; k < d; k++)
                s += M[IDX(i, k, d)] * M[IDX(j, k, d)];
            A[IDX(i, j, d)] = s;
            A[IDX(j, i, d)] = s;
        }
    for (int i = 0; i < d; i++)
        A[IDX(i, i, d)] += (float)d;
    free(M);
}

/* ---- Banachiewicz (row-by-row) ---------------------------------- */
#if defined(ALGO_BANACHIEWICZ)
static void cholesky(float *L, const float *A, int d)
{
    for (int i = 0; i < d; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++)
                sum += L[IDX(i, k, d)] * L[IDX(j, k, d)];
            if (i == j)
                L[IDX(i, j, d)] = sqrtf(A[IDX(i, j, d)] - sum);
            else
                L[IDX(i, j, d)] = (A[IDX(i, j, d)] - sum) / L[IDX(j, j, d)];
        }
    }
}
#endif

/* ---- Crout (column-by-column) ----------------------------------- */
#if defined(ALGO_CROUT)
static void cholesky(float *L, const float *A, int d)
{
    for (int j = 0; j < d; j++) {
        float sum = 0.0f;
        for (int k = 0; k < j; k++)
            sum += L[IDX(j, k, d)] * L[IDX(j, k, d)];
        L[IDX(j, j, d)] = sqrtf(A[IDX(j, j, d)] - sum);
        float ljj = L[IDX(j, j, d)];
        for (int i = j + 1; i < d; i++) {
            float s = 0.0f;
            for (int k = 0; k < j; k++)
                s += L[IDX(i, k, d)] * L[IDX(j, k, d)];
            L[IDX(i, j, d)] = (A[IDX(i, j, d)] - s) / ljj;
        }
    }
}
#endif

/* ----------------------------------------------------------------- */
int main(int argc, char **argv)
{
    assert(argc == 2);
    int d = atoi(argv[1]);
    assert(d > 0);

    size_t n2 = (size_t)d * d;
    float *A = (float *)calloc(n2, sizeof(float));
    float *L = (float *)calloc(n2, sizeof(float));
    assert(A && L);

    build_spd(A, d);

    /* ---- PAPI setup: native event for L1 data cache load misses --- */
    int ret = PAPI_library_init(PAPI_VER_CURRENT);
    if (ret != PAPI_VER_CURRENT)
        fprintf(stderr, "PAPI_library_init failed: %d\n", ret);

    int EventSet = PAPI_NULL;
    long long values[1] = { 0 };
    int event_code = 0;

    ret = PAPI_create_eventset(&EventSet);
    if (ret != PAPI_OK)
        fprintf(stderr, "PAPI_create_eventset failed: %s\n", PAPI_strerror(ret));

    /* Try preset first, fall back to native perf event */
    ret = PAPI_add_event(EventSet, PAPI_L1_DCM);
    if (ret != PAPI_OK) {
        ret = PAPI_event_name_to_code("perf::L1-DCACHE-LOAD-MISSES", &event_code);
        if (ret != PAPI_OK)
            fprintf(stderr, "PAPI_event_name_to_code failed: %s\n", PAPI_strerror(ret));
        else {
            ret = PAPI_add_event(EventSet, event_code);
            if (ret != PAPI_OK)
                fprintf(stderr, "PAPI_add_event(native) failed: %s\n", PAPI_strerror(ret));
        }
    }

    struct timespec t0, t1;

    ret = PAPI_start(EventSet);
    if (ret != PAPI_OK)
        fprintf(stderr, "PAPI_start failed: %s\n", PAPI_strerror(ret));

    clock_gettime(CLOCK_MONOTONIC, &t0);

    cholesky(L, A, d);

    clock_gettime(CLOCK_MONOTONIC, &t1);

    ret = PAPI_stop(EventSet, values);
    if (ret != PAPI_OK)
        fprintf(stderr, "PAPI_stop failed: %s\n", PAPI_strerror(ret));

    do_not_optimize(L);
    sink_f = L[0];

    double seconds = (double)(t1.tv_sec - t0.tv_sec)
                   + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double flops   = (double)d * (double)d * (double)d / 3.0;
    double gflops  = (flops / seconds) * 1e-9;

    long long l1_miss = values[0];

    printf("%s,%s,%d,%.15g,%lld,%.15g\n",
           LAYOUT_STR, ALGO_STR, d, gflops, l1_miss, seconds);

    free(L);
    free(A);
    return 0;
}
