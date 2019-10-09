#pragma once

#include "../../data/TensorTypes.hpp"
#include "../util/ConvolutionTools.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/Novelty.hpp"

#include <Eigen/Core>

namespace fluid {
namespace algorithm {

class NoveltySegmentation {

public:
  using ArrayXd = Eigen::ArrayXd;

  NoveltySegmentation(int maxKernelSize, int maxFilterSize)
      : mNovelty(maxKernelSize), mFilterBufferStorage(maxFilterSize){}

  void init(int kernelSize, double threshold, int filterSize, int nDims) {
    assert(kernelSize % 2);
    mThreshold = threshold;
    mFilterSize = filterSize;
    mNovelty.init(kernelSize, nDims);
    mFilterBuffer = mFilterBufferStorage.segment(0, mFilterSize);
    mFilterBuffer.setZero();
  }

  double processFrame(const RealVectorView input) {
    double novelty = mNovelty.processFrame(_impl::asEigen<Eigen::Array>(input));
    if (mFilterSize > 1) {
      mFilterBuffer.segment(0, mFilterSize - 1) =
          mFilterBuffer.segment(1, mFilterSize - 1);
    }
    mPeakBuffer.segment(0, 2) = mPeakBuffer.segment(1, 2);
    mFilterBuffer(mFilterSize - 1) = novelty;
    mPeakBuffer(2) = mFilterBuffer.mean();
    if (mPeakBuffer(1) > mPeakBuffer(0) && mPeakBuffer(1) > mPeakBuffer(2) &&
        mPeakBuffer(1) > mThreshold)
      return 1;
    else
      return 0;
  }

private:
  double mThreshold;
  int mFilterSize;
  ArrayXd mFilterBuffer;
  ArrayXd mFilterBufferStorage;
  ArrayXd mPeakBuffer{3};
  Novelty mNovelty;
};
} // namespace algorithm
} // namespace fluid
