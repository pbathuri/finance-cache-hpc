/*
 * mc_paths_papi.c
 *
 * PAPI-instrumented correlated MC path generation: z = L*eps, P paths.
 * Cholesky factorisation is outside the timed region.
 * Compile: -DROW_MAJOR or -DCOL_MAJOR
 * Usage:   ./mc_paths_<layout> <d> <P>
 * Output:  layout,kernel,d,P,gflops,l1_miss,seconds
 */

#include <assert.h>
#include <math.h>
#include <papi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* layout macros */
#if defined(ROW_MAJOR)
  #define IDX(i,j,n) ((i)*(n) + (j))
  #define LAYOUT_STR "rm"
#elif defined(COL_MAJOR)
  #define IDX(i,j,n) ((i) + (j)*(n))
  #define LAYOUT_STR "cm"
#else
  #error "Define ROW_MAJOR or COL_MAJOR"
#endif

void __attribute__((noinline)) do_not_optimize(float *_) { (void)_; }
static volatile float sink_f;

/* xorshift64 PRNG */
static unsigned long long xor_state = 0x12345678ABCDEF01ULL;

static inline unsigned long long xorshift64(void)
{
    unsigned long long x = xor_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    xor_state = x;
    return x;
}

static inline float uniform01(void)
{
    return (float)(xorshift64() & 0xFFFFFFFF) / 4294967296.0f;
}

static inline float randn(void)
{
    float u1 = uniform01();
    float u2 = uniform01();
    if (u1 < 1e-30f) u1 = 1e-30f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* build SPD and factor (not timed) */
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

static void cholesky_factor(float *L, const float *A, int d)
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

/* triangular matvec */
static inline void tri_matvec(const float *L, const float *eps,
                              float *z, int d)
{
    for (int i = 0; i < d; i++) {
        float s = 0.0f;
        for (int j = 0; j <= i; j++)
            s += L[IDX(i, j, d)] * eps[j];
        z[i] = s;
    }
}

int main(int argc, char **argv)
{
    assert(argc == 3);
    int d = atoi(argv[1]);
    int P = atoi(argv[2]);
    assert(d > 0 && P > 0);

    size_t n2 = (size_t)d * d;
    float *A   = (float *)calloc(n2, sizeof(float));
    float *L   = (float *)calloc(n2, sizeof(float));
    float *eps  = (float *)malloc((size_t)d * sizeof(float));
    float *z    = (float *)malloc((size_t)d * sizeof(float));
    float *acc  = (float *)calloc((size_t)d, sizeof(float));
    assert(A && L && eps && z && acc);

    /* setup */
    build_spd(A, d);
    cholesky_factor(L, A, d);
    free(A);

    /* PAPI init */
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

    /* timed: P paths */
    for (int p = 0; p < P; p++) {
        for (int i = 0; i < d; i++)
            eps[i] = randn();
        tri_matvec(L, eps, z, d);
        for (int i = 0; i < d; i++)
            acc[i] += z[i];
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    ret = PAPI_stop(EventSet, values);
    if (ret != PAPI_OK)
        fprintf(stderr, "PAPI_stop failed: %s\n", PAPI_strerror(ret));

    do_not_optimize(acc);
    sink_f = acc[0];

    double seconds = (double)(t1.tv_sec - t0.tv_sec)
                   + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double flops  = (double)P * (double)d * ((double)d + 1.0) / 2.0 * 2.0;
    double gflops = (flops / seconds) * 1e-9;

    long long l1_miss = values[0];

    printf("%s,mc_paths,%d,%d,%.15g,%lld,%.15g\n",
           LAYOUT_STR, d, P, gflops, l1_miss, seconds);

    free(acc);
    free(z);
    free(eps);
    free(L);
    return 0;
}
