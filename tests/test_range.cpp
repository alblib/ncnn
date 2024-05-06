// alblib is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 alblib <w890702@gmail.com>. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "testutil.h"

int main()
{
    ncnn::Mat start(1), end(1), step(1);
    start[0] = 2f;
    end[0] = 4f;
    step[0] = 1f;

    std::vector<ncnn::Mat> inputs = {start, end, step};

    ncnn::ParamDict pd;
    std::vector<ncnn::Mat> weights(0);

    test_layer("Range", pd, weights, inputs);
    return 0;
}