// Optimized CPU implementation of Y = ReLU(X * matrix + bias).
// Builds with the original Makefile: AVX2/FMA and pthreads, no extra libraries.
#include <stddef.h>
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <thread>
#include <vector>

namespace {

// Cache tiles. Each AVX2 microkernel computes up to 6 rows by 16 columns.
constexpr size_t ROW_BLOCK = 96;
constexpr size_t COL_BLOCK = 128;
constexpr size_t INNER_BLOCK = 256;
constexpr size_t MICRO_ROWS = 6;
constexpr size_t MICRO_COLS = 16;

// Also handles output columns that do not fill a complete SIMD tile.
void scalar_columns(size_t rows, size_t inner, size_t cols,
                    const float* matrix, const float* bias,
                    const float* input, float* output,
                    size_t begin, size_t end) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = begin; j < end; ++j) {
            float value = 0.0f;
            for (size_t k = 0; k < inner; ++k)
                value += input[i * inner + k] * matrix[k * cols + j];
            value += bias[j];
            output[i * cols + j] = value < 0.0f ? 0.0f : value;
        }
    }
}

template <int Rows>
inline void microkernel(size_t count, const float* input, size_t input_stride,
                        const float* packed, float* output, size_t output_stride,
                        const float* bias, bool first, bool last) {
    __m256 low[Rows];
    __m256 high[Rows];

    // Only the first inner block starts from zero. Later blocks accumulate.
    #pragma GCC unroll 6
    for (int r = 0; r < Rows; ++r) {
        low[r] = first ? _mm256_setzero_ps()
                       : _mm256_loadu_ps(output + r * output_stride);
        high[r] = first ? _mm256_setzero_ps()
                        : _mm256_loadu_ps(output + r * output_stride + 8);
    }

    for (size_t k = 0; k < count; ++k) {
        const __m256 weights_low = _mm256_loadu_ps(packed + k * MICRO_COLS);
        const __m256 weights_high = _mm256_loadu_ps(packed + k * MICRO_COLS + 8);
        #pragma GCC unroll 6
        for (int r = 0; r < Rows; ++r) {
            const __m256 x = _mm256_set1_ps(input[r * input_stride + k]);
            low[r] = _mm256_fmadd_ps(x, weights_low, low[r]);
            high[r] = _mm256_fmadd_ps(x, weights_high, high[r]);
        }
    }

    // Apply bias and ReLU once, after the ENTIRE dot product is complete.
    if (last) {
        const __m256 bias_low = _mm256_loadu_ps(bias);
        const __m256 bias_high = _mm256_loadu_ps(bias + 8);
        const __m256 zero = _mm256_setzero_ps();
        #pragma GCC unroll 6
        for (int r = 0; r < Rows; ++r) {
            low[r] = _mm256_max_ps(zero, _mm256_add_ps(low[r], bias_low));
            high[r] = _mm256_max_ps(zero, _mm256_add_ps(high[r], bias_high));
        }
    }

    #pragma GCC unroll 6
    for (int r = 0; r < Rows; ++r) {
        _mm256_storeu_ps(output + r * output_stride, low[r]);
        _mm256_storeu_ps(output + r * output_stride + 8, high[r]);
    }
}

struct Work {
    size_t rows, inner, cols, vector_cols;
    const float *matrix, *bias, *input;
    float* output;
    std::atomic<size_t> next_panel{0};
};

void worker(Work& work) {
    // Private packing buffer: 2 MiB for the supplied 4096-feature layer.
    float* packed = static_cast<float*>(
        std::malloc(work.inner * COL_BLOCK * sizeof(float)));
    const size_t panel_count =
        (work.vector_cols + COL_BLOCK - 1) / COL_BLOCK;

    for (;;) {
        const size_t panel = work.next_panel.fetch_add(1, std::memory_order_relaxed);
        if (panel >= panel_count) break;
        const size_t col = panel * COL_BLOCK;
        const size_t width = std::min(COL_BLOCK, work.vector_cols - col);

        if (!packed) {
            // Keep the result correct even if the packing allocation fails.
            scalar_columns(work.rows, work.inner, work.cols, work.matrix,
                           work.bias, work.input, work.output, col, col + width);
            continue;
        }

        // Pack this weight panel ONCE, then reuse it for all input examples.
        // Layout per inner block: [16-column tile][k][16 weights].
        for (size_t kb = 0; kb < work.inner; kb += INNER_BLOCK) {
            const size_t depth = std::min(INNER_BLOCK, work.inner - kb);
            for (size_t j = 0; j < width; j += MICRO_COLS) {
                float* destination = packed + kb * COL_BLOCK + j * depth;
                for (size_t k = 0; k < depth; ++k) {
                    std::memcpy(destination + k * MICRO_COLS,
                                work.matrix + (kb + k) * work.cols + col + j,
                                MICRO_COLS * sizeof(float));
                }
            }
        }

        for (size_t rb = 0; rb < work.rows; rb += ROW_BLOCK) {
            const size_t row_end = std::min(rb + ROW_BLOCK, work.rows);
            for (size_t kb = 0; kb < work.inner; kb += INNER_BLOCK) {
                const size_t depth = std::min(INNER_BLOCK, work.inner - kb);
                const bool first = kb == 0;
                const bool last = kb + depth == work.inner;

                for (size_t j = 0; j < width; j += MICRO_COLS) {
                    const float* weights = packed + kb * COL_BLOCK + j * depth;
                    const float* bias = work.bias + col + j;
                    size_t row = rb;
                    for (; row + MICRO_ROWS <= row_end; row += MICRO_ROWS) {
                        microkernel<6>(depth, work.input + row * work.inner + kb,
                                       work.inner, weights,
                                       work.output + row * work.cols + col + j,
                                       work.cols, bias, first, last);
                    }

                    if (row == row_end) continue;

                    // A final partial row tile must not read beyond the input.
                    const float* x = work.input + row * work.inner + kb;
                    float* y = work.output + row * work.cols + col + j;
                    switch (row_end - row) {
                    case 5: microkernel<5>(depth, x, work.inner, weights, y,
                                          work.cols, bias, first, last); break;
                    case 4: microkernel<4>(depth, x, work.inner, weights, y,
                                          work.cols, bias, first, last); break;
                    case 3: microkernel<3>(depth, x, work.inner, weights, y,
                                          work.cols, bias, first, last); break;
                    case 2: microkernel<2>(depth, x, work.inner, weights, y,
                                          work.cols, bias, first, last); break;
                    case 1: microkernel<1>(depth, x, work.inner, weights, y,
                                          work.cols, bias, first, last); break;
                    default: break;
                    }
                }
            }
        }
    }
    std::free(packed);
}

} // namespace

void fc_layer(size_t data_cnt, size_t input_dim, size_t output_dim,
              float* matrix, float* bias, float* input, float* output,
              int threads) {
    if (data_cnt == 0 || output_dim == 0) return;

    const size_t vector_cols = output_dim / MICRO_COLS * MICRO_COLS;
    if (input_dim == 0 || vector_cols == 0) {
        scalar_columns(data_cnt, input_dim, output_dim, matrix, bias,
                       input, output, 0, output_dim);
        return;
    }

    Work work;
    work.rows = data_cnt;
    work.inner = input_dim;
    work.cols = output_dim;
    work.vector_cols = vector_cols;
    work.matrix = matrix;
    work.bias = bias;
    work.input = input;
    work.output = output;

    // Threads own disjoint output columns, so no output locks are necessary.
    const size_t panel_count = (vector_cols + COL_BLOCK - 1) / COL_BLOCK;
    const size_t worker_count = std::min(
        panel_count, static_cast<size_t>(threads > 0 ? threads : 1));
    std::vector<std::thread> pool;
    try {
        pool.reserve(worker_count - 1);
        for (size_t t = 1; t < worker_count; ++t)
            pool.emplace_back([&work]() { worker(work); });
    } catch (const std::exception&) {
        // If thread creation runs out of resources, use the available workers.
    }

    worker(work); // The calling thread also performs useful work.
    for (std::thread& thread : pool) thread.join();

    scalar_columns(data_cnt, input_dim, output_dim, matrix, bias,
                   input, output, vector_cols, output_dim);
}
