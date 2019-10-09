
#pragma once

#include "../../data/TensorTypes.hpp"
#include "../util/AlgorithmUtils.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "Windows.hpp"

#include <Eigen/Eigen>

namespace fluid {
namespace algorithm {


class SpectralShape {

using ArrayXd = Eigen::ArrayXd;

public:

  SpectralShape(size_t maxFrame) : mMagBuffer(maxFrame), mOutputBuffer(7) {}

  void processFrame(Eigen::Ref<ArrayXd> in) {
    double const epsilon = std::numeric_limits<double>::epsilon();
    ArrayXd x = in.max(epsilon);
    int size = x.size();
    double xSum = x.sum();
    ArrayXd xSquare = x.square();
    ArrayXd lin = ArrayXd::LinSpaced(size, 0, size - 1);
    double centroid = (x * lin).sum() / xSum;
    double spread = (x * (lin - centroid).square()).sum() / xSum;
    double skewness =
        (x * (lin - centroid).pow(3)).sum() / (spread * sqrt(spread) * xSum);
    double kurtosis =
        (x * (lin - centroid).pow(4)).sum() / (spread * spread * xSum);
    double flatness = exp(x.log().mean()) / x.mean();
    double rolloff = size - 1;
    double cumSum = 0;
    double target = 0.95 *  xSquare.sum();
    for (int i = 0; cumSum <= target && i < size; i++) {
      cumSum += xSquare(i);
      if (cumSum > target) {
        rolloff = i - (cumSum - target) / xSquare(i);
        break;
      }
    }
    double crest = x.maxCoeff() / sqrt(x.square().mean());

    mOutputBuffer(0) = centroid;
    mOutputBuffer(1) = sqrt(spread);
    mOutputBuffer(2) = skewness;
    mOutputBuffer(3) = kurtosis;
    mOutputBuffer(4) = rolloff;
    mOutputBuffer(5) = 20 * std::log10(std::max(flatness, epsilon));
    mOutputBuffer(6) = 20 * std::log10(std::max(crest, epsilon));
  }

  void processFrame(const RealVector &input, RealVectorView output) {
    assert(output.size() == 7); // TODO
    ArrayXd in = _impl::asEigen<Eigen::Array>(input);
    processFrame(in);
    output = _impl::asFluid(mOutputBuffer);
  }

private:
  ArrayXd mMagBuffer;
  ArrayXd mOutputBuffer;
};

}; // namespace algorithm
}; // namespace fluid
