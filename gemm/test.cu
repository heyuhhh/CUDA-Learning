#include <cstdio>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

int main() {
    float *a = new float[10];
    for (int i = 0; i < 10; i++) {
        a[i] = i;
    }

    float4 &f = reinterpret_cast<float4*>(a)[1];
    float4 &g = reinterpret_cast<float4*>(a)[0];
    g = f;
    printf("%f\n", g.x);
    for (int i = 0; i < 10; i++) {
        printf("%f ", a[i]);
    }
    return 0;
}