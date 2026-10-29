#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}



/**
 * Visit every element of a strided (non-compact) view in row-major order, calling
 * fn(cnt, idx) where cnt is the position in the equivalent compact array and idx is the
 * position in the underlying strided memory.  Uses a vector of per-dimension indices that
 * is incremented like an odometer, so it works for any number of dimensions.
 */
template <typename F>
void ForEachStrided(const std::vector<int32_t>& shape, const std::vector<int32_t>& strides,
                    size_t offset, F fn) {
  size_t ndim = shape.size();
  size_t total = 1;
  for (size_t d = 0; d < ndim; d++) total *= shape[d];
  if (total == 0) return;

  std::vector<int32_t> index(ndim, 0);
  size_t pos = offset;
  for (size_t cnt = 0; cnt < total; cnt++) {
    fn(cnt, pos);
    // increment the last dimension, carrying into earlier ones as needed
    for (int d = (int)ndim - 1; d >= 0; d--) {
      index[d]++;
      pos += strides[d];
      if (index[d] < shape[d]) break;
      pos -= (size_t)strides[d] * shape[d];
      index[d] = 0;
    }
  }
}

void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN SOLUTION
  ForEachStrided(shape, strides, offset,
                 [&](size_t cnt, size_t idx) { out->ptr[cnt] = a.ptr[idx]; });
  /// END SOLUTION
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  ForEachStrided(shape, strides, offset,
                 [&](size_t cnt, size_t idx) { out->ptr[idx] = a.ptr[cnt]; });
  /// END SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
  ForEachStrided(shape, strides, offset, [&](size_t cnt, size_t idx) { out->ptr[idx] = val; });
  /// END SOLUTION
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}


/**
 * In the code the follows, use the above template to create analogous element-wise
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

template <typename Op>
void EwiseOp(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, Op op) {
  for (size_t i = 0; i < a.size; i++) out->ptr[i] = op(a.ptr[i], b.ptr[i]);
}

template <typename Op>
void ScalarOp(const AlignedArray& a, scalar_t val, AlignedArray* out, Op op) {
  for (size_t i = 0; i < a.size; i++) out->ptr[i] = op(a.ptr[i], val);
}

template <typename Op>
void UnaryOp(const AlignedArray& a, AlignedArray* out, Op op) {
  for (size_t i = 0; i < a.size; i++) out->ptr[i] = op(a.ptr[i]);
}

#define DEFINE_BINARY_OP(NAME, EXPR)                                                void Ewise##NAME(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {     EwiseOp(a, b, out, [](scalar_t x, scalar_t y) -> scalar_t { return EXPR; });    }                                                                                 void Scalar##NAME(const AlignedArray& a, scalar_t val, AlignedArray* out) {         ScalarOp(a, val, out, [](scalar_t x, scalar_t y) -> scalar_t { return EXPR; });   }

#define DEFINE_UNARY_OP(NAME, EXPR)                                  void Ewise##NAME(const AlignedArray& a, AlignedArray* out) {         UnaryOp(a, out, [](scalar_t x) -> scalar_t { return EXPR; });    }

DEFINE_BINARY_OP(Mul, x * y)
DEFINE_BINARY_OP(Div, x / y)
DEFINE_BINARY_OP(Maximum, std::max(x, y))
DEFINE_BINARY_OP(Eq, x == y ? 1.0f : 0.0f)
DEFINE_BINARY_OP(Ge, x >= y ? 1.0f : 0.0f)

void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarOp(a, val, out, [](scalar_t x, scalar_t y) -> scalar_t { return std::pow(x, y); });
}

DEFINE_UNARY_OP(Log, std::log(x))
DEFINE_UNARY_OP(Exp, std::exp(x))
DEFINE_UNARY_OP(Tanh, std::tanh(x))


void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < m; i++) {
    scalar_t* out_row = out->ptr + i * p;
    for (size_t j = 0; j < p; j++) out_row[j] = 0;
    // i-k-j order so the inner loop walks b and out contiguously
    for (size_t k = 0; k < n; k++) {
      scalar_t a_ik = a.ptr[i * n + k];
      const scalar_t* b_row = b.ptr + k * p;
      for (size_t j = 0; j < p; j++) out_row[j] += a_ik * b_row[j];
    }
  }
  /// END SOLUTION
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  for (size_t i = 0; i < TILE; i++) {
    for (size_t k = 0; k < TILE; k++) {
      float a_ik = a[i * TILE + k];
      for (size_t j = 0; j < TILE; j++) {
        out[i * TILE + j] += a_ik * b[k * TILE + j];
      }
    }
  }
  /// END SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN SOLUTION
  const size_t TILE_SIZE = TILE * TILE;
  size_t mt = m / TILE, nt = n / TILE, pt = p / TILE;
  for (size_t i = 0; i < out->size; i++) out->ptr[i] = 0;
  for (size_t i = 0; i < mt; i++) {
    for (size_t j = 0; j < pt; j++) {
      float* out_tile = out->ptr + (i * pt + j) * TILE_SIZE;
      for (size_t k = 0; k < nt; k++) {
        AlignedDot(a.ptr + (i * nt + k) * TILE_SIZE, b.ptr + (k * pt + j) * TILE_SIZE, out_tile);
      }
    }
  }
  /// END SOLUTION
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < out->size; i++) {
    const scalar_t* block = a.ptr + i * reduce_size;
    scalar_t m = block[0];
    for (size_t j = 1; j < reduce_size; j++) m = std::max(m, block[j]);
    out->ptr[i] = m;
  }
  /// END SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < out->size; i++) {
    const scalar_t* block = a.ptr + i * reduce_size;
    scalar_t sum = 0;
    for (size_t j = 0; j < reduce_size; j++) sum += block[j];
    out->ptr[i] = sum;
  }
  /// END SOLUTION
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
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
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
