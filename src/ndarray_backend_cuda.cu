#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides
__device__ size_t StridedIndex(size_t gid, const CudaVec& shape, const CudaVec& strides,
                               size_t offset) {
  size_t idx = offset;
  for (int d = (int)shape.size - 1; d >= 0; d--) {
    idx += (gid % shape.data[d]) * strides.data[d];
    gid /= shape.data[d];
  }
  return idx;
}

size_t ShapeSize(const std::vector<int32_t>& shape) {
  size_t size = 1;
  for (int32_t s : shape) size *= s;
  return size;
}

__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of a array
   *   offset: offset of a array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid < size) out[gid] = a[StridedIndex(gid, shape, strides, offset)];
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}


__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                                   CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[StridedIndex(gid, shape, strides, offset)] = a[gid];
}

__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                                    CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[StridedIndex(gid, shape, strides, offset)] = val;
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  size_t size = ShapeSize(shape);
  CudaDims dim = CudaOneDim(size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, size, VecToCuda(shape),
                                              VecToCuda(strides), offset);
  /// END SOLUTION
}



void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                               VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

#define DEFINE_EWISE_OP(NAME, EXPR)                                                        \
  __global__ void Ewise##NAME##Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, \
                                      size_t size) {                                       \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                                    \
    if (gid < size) {                                                                      \
      scalar_t x = a[gid], y = b[gid];                                                     \
      out[gid] = (EXPR);                                                                   \
    }                                                                                      \
  }                                                                                        \
  void Ewise##NAME(const CudaArray& a, const CudaArray& b, CudaArray* out) {               \
    CudaDims dim = CudaOneDim(out->size);                                                  \
    Ewise##NAME##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);       \
  }

#define DEFINE_SCALAR_OP(NAME, EXPR)                                                 \
  __global__ void Scalar##NAME##Kernel(const scalar_t* a, scalar_t y, scalar_t* out, \
                                       size_t size) {                                \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                              \
    if (gid < size) {                                                                \
      scalar_t x = a[gid];                                                           \
      out[gid] = (EXPR);                                                             \
    }                                                                                \
  }                                                                                  \
  void Scalar##NAME(const CudaArray& a, scalar_t val, CudaArray* out) {              \
    CudaDims dim = CudaOneDim(out->size);                                            \
    Scalar##NAME##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);  \
  }

#define DEFINE_BINARY_OP(NAME, EXPR) \
  DEFINE_EWISE_OP(NAME, EXPR)        \
  DEFINE_SCALAR_OP(NAME, EXPR)

#define DEFINE_UNARY_OP(NAME, EXPR)                                                    \
  __global__ void Ewise##NAME##Kernel(const scalar_t* a, scalar_t* out, size_t size) { \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                                \
    if (gid < size) {                                                                  \
      scalar_t x = a[gid];                                                             \
      out[gid] = (EXPR);                                                               \
    }                                                                                  \
  }                                                                                    \
  void Ewise##NAME(const CudaArray& a, CudaArray* out) {                               \
    CudaDims dim = CudaOneDim(out->size);                                              \
    Ewise##NAME##Kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);          \
  }

DEFINE_BINARY_OP(Mul, x * y)
DEFINE_BINARY_OP(Div, x / y)
DEFINE_BINARY_OP(Maximum, fmaxf(x, y))
DEFINE_BINARY_OP(Eq, x == y ? 1.0f : 0.0f)
DEFINE_BINARY_OP(Ge, x >= y ? 1.0f : 0.0f)
DEFINE_SCALAR_OP(Power, powf(x, y))

DEFINE_UNARY_OP(Log, logf(x))
DEFINE_UNARY_OP(Exp, expf(x))
DEFINE_UNARY_OP(Tanh, tanhf(x))


////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

// Each block computes an MM_BM x MM_BN tile of the output, stepping MM_BK along the shared
// dimension.  Each thread accumulates an MM_TM x MM_TN register tile, so a block has
// (MM_BM / MM_TM) * (MM_BN / MM_TN) threads.
#define MM_BM 64
#define MM_BN 64
#define MM_BK 16
#define MM_TM 4
#define MM_TN 4
#define MM_THREADS ((MM_BM / MM_TM) * (MM_BN / MM_TN))

__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M,
                             uint32_t N, uint32_t P) {
  // the A tile is stored transposed so the inner loop reads both tiles along rows
  __shared__ scalar_t a_tile[MM_BK][MM_BM];
  __shared__ scalar_t b_tile[MM_BK][MM_BN];

  const int tid = threadIdx.x;
  const int tx = tid % (MM_BN / MM_TN);
  const int ty = tid / (MM_BN / MM_TN);
  const size_t row0 = (size_t)blockIdx.y * MM_BM;
  const size_t col0 = (size_t)blockIdx.x * MM_BN;

  scalar_t acc[MM_TM][MM_TN] = {{0}};
  scalar_t a_reg[MM_TM], b_reg[MM_TN];

  for (size_t k0 = 0; k0 < N; k0 += MM_BK) {
    // cooperative fetch of both tiles into shared memory, zero-padding out-of-range entries
    for (int l = tid; l < MM_BM * MM_BK; l += MM_THREADS) {
      int r = l / MM_BK, c = l % MM_BK;
      size_t gr = row0 + r, gc = k0 + c;
      a_tile[c][r] = (gr < M && gc < N) ? a[gr * N + gc] : 0;
    }
    for (int l = tid; l < MM_BK * MM_BN; l += MM_THREADS) {
      int r = l / MM_BN, c = l % MM_BN;
      size_t gr = k0 + r, gc = col0 + c;
      b_tile[r][c] = (gr < N && gc < P) ? b[gr * P + gc] : 0;
    }
    __syncthreads();

    for (int kk = 0; kk < MM_BK; kk++) {
      for (int i = 0; i < MM_TM; i++) a_reg[i] = a_tile[kk][ty * MM_TM + i];
      for (int j = 0; j < MM_TN; j++) b_reg[j] = b_tile[kk][tx * MM_TN + j];
      for (int i = 0; i < MM_TM; i++)
        for (int j = 0; j < MM_TN; j++) acc[i][j] += a_reg[i] * b_reg[j];
    }
    __syncthreads();
  }

  for (int i = 0; i < MM_TM; i++) {
    size_t r = row0 + ty * MM_TM + i;
    if (r >= M) break;
    for (int j = 0; j < MM_TN; j++) {
      size_t c = col0 + tx * MM_TN + j;
      if (c < P) out[r * P + c] = acc[i][j];
    }
  }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  dim3 grid((P + MM_BN - 1) / MM_BN, (M + MM_BM - 1) / MM_BM, 1);
  dim3 block(MM_THREADS, 1, 1);
  MatmulKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t size,
                                size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    const scalar_t* block = a + gid * reduce_size;
    scalar_t m = block[0];
    for (size_t j = 1; j < reduce_size; j++) m = fmaxf(m, block[j]);
    out[gid] = m;
  }
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t size,
                                size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    const scalar_t* block = a + gid * reduce_size;
    scalar_t sum = 0;
    for (size_t j = 0; j < reduce_size; j++) sum += block[j];
    out[gid] = sum;
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}



void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
