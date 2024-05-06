#include "range.h"
#include <cmath>

namespace ncnn {

Range::Range()
{
    one_blob_only = false;
    support_inplace = false;
}

bool is_same_shape(const Mat& a, const Mat& b)
{
    if (a.dims != b.dims) return false;
    if (a.w != b.w) return false;
    if (a.h != b.h) return false;
    if (a.d != b.d) return false;
    if (a.c != b.c) return false;
    return true;
}

int Range::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& start = bottom_blobs[0];
    const Mat& limit = bottom_blobs[1];
    const Mat& delta = bottom_blobs[2];

    const float* start_elem_ptr = start.data;
    const float* limit_elem_ptr = limit.data;
    const float* delta_elem_ptr = delta.data;

    const float& start_elem = *start_elem_ptr;
    const float& limit_elem = *limit_elem_ptr;
    const float& delta_elem = *delta_elem_ptr;
    const long size = std::ceil((limit_elem - start_elem) / delta_elem);

    Mat& top_blob = top_blobs[0];
    top_blob.create(size, 4u, opt.blob_allocator);

    long i = 0; float elem = start_elem;
    while (i < size)
    {
        top_blob[i] = elem;
        ++i; elem += delta_elem;
    }
    return 0;
}

} // namespace ncnn