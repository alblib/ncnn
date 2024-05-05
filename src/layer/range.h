#ifndef LAYER_RANGE_H
#define LAYER_RANGE_H

#include "layer.h"

namespace ncnn {

class Range : public Layer
{
public:
    Range();

    //virtual int load_param(const ParamDict& pd);

    using Layer::forward;
    using Layer::forward_inplace;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
};

} // namespace ncnn

#endif // LAYER_RANGE_H