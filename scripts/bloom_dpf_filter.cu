#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>

namespace {

struct Matrix {
    const int* row;
    const int* col;
    const float* value;
};

template <typename T>
struct Buffer {
    T* ptr = nullptr;
    size_t capacity = 0;

    cudaError_t resize(size_t n) {
        if (n <= capacity) return cudaSuccess;
        cudaError_t status = cudaFree(ptr);
        if (status != cudaSuccess) return status;
        ptr = nullptr;
        capacity = 0;
        status = cudaMalloc(&ptr, n * sizeof(T));
        if (status == cudaSuccess) capacity = n;
        return status;
    }
};

struct Workspace {
    Buffer<float> source, delta, result, input_gradient, x, x_gradient, x_position, weights, coefficients;
    Buffer<int> integers;
    Buffer<double> weight_gradient, position_gradient;
};

constexpr int block = 256;

__device__ float row_dot(Matrix m, int row, const float* src, int stride) {
    float value = 0;
    for (int k = m.row[row]; k < m.row[row + 1]; ++k)
        value += m.value[k] * src[m.col[k] * stride];
    return value;
}

__device__ double warp_sum(double value) {
    for (int offset = 16; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}

__device__ void accumulate(double* dest, double value) {
    value = warp_sum(value);
    if ((threadIdx.x & 31) == 0) atomicAdd(dest, value);
}

__global__ void along_x(const float* src, float* dest, Matrix m, int height, int sw, int dw,
                        int sc, int dc, bool add) {
    const int plane = blockIdx.y;
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= height * dw) return;
    const int input_plane = (plane / dc) * sc + (sc == 1 ? 0 : plane % dc);
    const float* row = src + (input_plane * height + pixel / dw) * sw;
    const float value = row_dot(m, pixel % dw, row, 1);
    if (add) dest[plane * height * dw + pixel] += value;
    else dest[plane * height * dw + pixel] = value;
}

__global__ void along_y(const float* src, float* dest, Matrix m, const float* weights,
                        int sh, int dh, int width, int channels, bool add) {
    const int plane = blockIdx.y;
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= dh * width) return;
    const float* column = src + plane * sh * width + pixel % width;
    const float value = row_dot(m, pixel / width, column, width) * weights[plane % channels];
    if (add) dest[plane * dh * width + pixel] += value;
    else dest[plane * dh * width + pixel] = value;
}

__global__ void y_derivatives(const float* x, const float* delta, Matrix m, Matrix md,
                              const float* weights, double* weight_gradient,
                              double* position_gradient, int sh, int dh, int width, int channels) {
    const int plane = blockIdx.y;
    const int channel = plane % channels;
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    double dw = 0;
    double dp = 0;
    if (pixel < dh * width) {
        const float* column = x + plane * sh * width + pixel % width;
        const float d = delta[plane * dh * width + pixel];
        dw = d * row_dot(m, pixel / width, column, width);
        dp = (d * weights[channel]) * row_dot(md, pixel / width, column, width);
    }
    accumulate(weight_gradient + channel, dw);
    accumulate(position_gradient, dp);
}

__global__ void dot(const float* x, const float* gradient, double* result, int count) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    accumulate(result, pixel < count ? x[pixel] * gradient[pixel] : 0);
}

int blocks(int n) { return (n + block - 1) / block; }

} // namespace

extern "C" __declspec(dllexport)
int dpf_filter(
    const float* image, const float* derivative, const float* weights,
    int phases, int source_channels, int channels, int sh, int sw, int dh, int dw, int taps,
    const int* plans, const int* integers, int integer_count, const float* coefficients, int coefficient_count,
    float* output, float* input_gradient, double* weight_gradient, double* position_gradient) {
    static Workspace gpu;
    const int planes = phases * channels;
    const int inputs = phases * source_channels * sh * sw;
    const int outputs = planes * dh * dw;
    const int scratch = planes * sh * dw;
    const bool gradient = derivative != nullptr;
    cudaError_t status;

#define CUDA_CHECK(expr) do { status = (expr); if (status != cudaSuccess) return int(status); } while (false)
#define UPLOAD(buffer, cpu, count) \
    CUDA_CHECK(gpu.buffer.resize(count)); \
    CUDA_CHECK(cudaMemcpyAsync(gpu.buffer.ptr, cpu, (count) * sizeof(*(cpu)), cudaMemcpyHostToDevice))

    UPLOAD(source, image, inputs);
    UPLOAD(weights, weights, taps * channels);
    UPLOAD(integers, integers, integer_count);
    UPLOAD(coefficients, coefficients, coefficient_count);
    CUDA_CHECK(gpu.x.resize(scratch));
    if (gradient) {
        UPLOAD(delta, derivative, outputs);
        CUDA_CHECK(gpu.x_gradient.resize(scratch));
        CUDA_CHECK(gpu.x_position.resize(scratch));
        CUDA_CHECK(gpu.weight_gradient.resize(taps * channels));
        CUDA_CHECK(gpu.position_gradient.resize(taps * 2));
        CUDA_CHECK(cudaMemsetAsync(gpu.weight_gradient.ptr, 0, taps * channels * sizeof(double)));
        CUDA_CHECK(cudaMemsetAsync(gpu.position_gradient.ptr, 0, taps * 2 * sizeof(double)));
        if (input_gradient) {
            CUDA_CHECK(gpu.input_gradient.resize(planes * sh * sw));
            CUDA_CHECK(cudaMemsetAsync(gpu.input_gradient.ptr, 0, planes * sh * sw * sizeof(float)));
        }
    } else {
        CUDA_CHECK(gpu.result.resize(outputs));
        CUDA_CHECK(cudaMemsetAsync(gpu.result.ptr, 0, outputs * sizeof(float)));
    }

    const int matrices = gradient ? 6 : 2;
    for (int tap = 0; tap < taps; ++tap) {
        const auto matrix = [&](int axis) {
            const int* plan = plans + (tap * matrices + axis) * 3;
            return Matrix{gpu.integers.ptr + plan[0], gpu.integers.ptr + plan[1], gpu.coefficients.ptr + plan[2]};
        };
        const float* w = gpu.weights.ptr + tap * channels;
        const dim3 x_grid(blocks(sh * dw), planes);
        const dim3 y_grid(blocks(dh * dw), planes);
        along_x<<<x_grid, block>>>(gpu.source.ptr, gpu.x.ptr, matrix(0), sh, sw, dw, source_channels, channels, false);
        if (!gradient) {
            along_y<<<y_grid, block>>>(gpu.x.ptr, gpu.result.ptr, matrix(1), w, sh, dh, dw, channels, true);
            continue;
        }
        double* dp = gpu.position_gradient.ptr + tap * 2;
        y_derivatives<<<y_grid, block>>>(gpu.x.ptr, gpu.delta.ptr, matrix(1), matrix(5), w,
                                       gpu.weight_gradient.ptr + tap * channels, dp + 1, sh, dh, dw, channels);
        along_y<<<x_grid, block>>>(gpu.delta.ptr, gpu.x_gradient.ptr, matrix(3), w, dh, sh, dw, channels, false);
        along_x<<<x_grid, block>>>(gpu.source.ptr, gpu.x_position.ptr, matrix(4), sh, sw, dw, source_channels, channels, false);
        dot<<<blocks(scratch), block>>>(gpu.x_position.ptr, gpu.x_gradient.ptr, dp, scratch);
        if (input_gradient)
            along_x<<<dim3(blocks(sh * sw), planes), block>>>(gpu.x_gradient.ptr, gpu.input_gradient.ptr,
                                                             matrix(2), sh, dw, sw, channels, channels, true);
    }

    CUDA_CHECK(cudaPeekAtLastError());
#define DOWNLOAD(cpu, buffer, count) \
    CUDA_CHECK(cudaMemcpyAsync(cpu, gpu.buffer.ptr, (count) * sizeof(*(cpu)), cudaMemcpyDeviceToHost))
    if (gradient) {
        DOWNLOAD(weight_gradient, weight_gradient, taps * channels);
        DOWNLOAD(position_gradient, position_gradient, taps * 2);
        if (input_gradient) {
            DOWNLOAD(input_gradient, input_gradient, planes * sh * sw);
        }
    } else {
        DOWNLOAD(output, result, outputs);
    }
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    return 0;
}

extern "C" __declspec(dllexport)
const char* dpf_error(int code) { return cudaGetErrorString(cudaError_t(code)); }
