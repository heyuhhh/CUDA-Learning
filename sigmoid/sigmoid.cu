// sigmoid: x: N, y: N
// y[i] = 1 / (1 + exp(-x_i))
__global__ void sigmoid(float* x, float* y, const int N) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx < N) {
        float4 reg_x = reinterpret_cast<float4*>(x + idx)[0];
        float4 reg_y;
        if (idx + 0 < N) {
            reg_y.x = 1.0 / (1 + expf(-reg_x.x));
        } 
        if (idx + 1 < N) {
            reg_y.y = 1.0 / (1 + expf(-reg_x.y));
        }
        if (idx + 2 < N) {
            reg_y.z = 1.0 / (1 + expf(-reg_x.z));
        }
        if (idx + 3 < N) {
            reg_y.w = 1.0 / (1 + expf(-reg_x.w));
        }
        reinterpret_cast<float4*>(y + idx)[0] = reg_y;
    }
}