#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>


namespace nb = nanobind;
using namespace nb::literals;


template <typename Scalar>
__global__ void double_arr_kernel(Scalar* out, const Scalar* in, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = 2 * in[i];
    }
}


template <typename Scalar>
void double_arr(nb::ndarray<Scalar, nb::ndim<1>, nb::device::cuda> outarr,
            nb::ndarray<const Scalar, nb::ndim<1>, nb::device::cuda> inarr) {

    size_t size = inarr.size();
    size_t block_size = 256;
    size_t num_blocks = (size + block_size - 1) / block_size;

    double_arr_kernel<<<num_blocks, block_size>>>(outarr.data(), inarr.data(), size);
}


NB_MODULE(cuda, m) {
    m.def("double_arr",
        &double_arr<float>,
        "outarr"_a.noconvert(),
        "inarr"_a.noconvert()
        );
}
