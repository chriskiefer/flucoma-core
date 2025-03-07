/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
#pragma once

#include "../common/AudioClient.hpp"
#include "../common/BufferedProcess.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/OnsetSegmentation.hpp"
#include "../../data/TensorTypes.hpp"
#include <tuple>

namespace fluid {
namespace client {
namespace onsetslice {

using algorithm::OnsetSegmentation;

enum OnsetParamIndex {
  kFunction,
  kThreshold,
  kDebounce,
  kFilterSize,
  kFrameDelta,
  kFFT,
  kMaxFFTSize
};

constexpr auto OnsetSliceParams = defineParameters(
    EnumParam("metric", "Spectral Change Metric", 0, "Energy",
              "High Frequency Content", "Spectral Flux",
              "Modified Kullback-Leibler", "Itakura-Saito", "Cosine",
              "Phase Deviation", "Weighted Phase Deviation", "Complex Domain",
              "Rectified Complex Domain"),
    FloatParam("threshold", "Threshold", 0.5, Min(0)),
    LongParam("minSliceLength", "Minimum Length of Slice", 2, Min(0)),
    LongParam("filterSize", "Filter Size", 5, Min(1), Odd(), Max(101)),
    LongParam("frameDelta", "Frame Delta", 0, Min(0)),
    FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4),
                           PowerOfTwo{}));

class OnsetSliceClient : public FluidBaseClient, public AudioIn, public AudioOut
{
public:
  using ParamDescType = decltype(OnsetSliceParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return OnsetSliceParams; }

  OnsetSliceClient(ParamSetViewType& p)
      : mParams{p}, mAlgorithm{get<kMaxFFTSize>()}
  {
    audioChannelsIn(1);
    audioChannelsOut(1);
    setInputLabels({"audio input"});
    setOutputLabels({"1 when slice detected, 0 otherwise"});
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    using algorithm::OnsetSegmentation;
    using std::size_t;

    if (!input[0].data() || !output[0].data()) return;

    index hostVecSize = input[0].size();
    index totalWindow = get<kFFT>().winSize();
    if (get<kFunction>() > 1 && get<kFunction>() < 5)
      totalWindow += get<kFrameDelta>();
    if (mBufferParamsTracker.changed(hostVecSize, get<kFFT>().winSize(),
                                     get<kFrameDelta>()))
    {
      mBufferedProcess.hostSize(hostVecSize);
      mBufferedProcess.maxSize(totalWindow, totalWindow,
                               FluidBaseClient::audioChannelsIn(),
                               FluidBaseClient::audioChannelsOut());
    }
    if (mParamsTracker.changed(get<kFFT>().fftSize(), get<kFFT>().winSize()))
    {
      mAlgorithm.init(get<kFFT>().winSize(), get<kFFT>().fftSize(),
                      get<kFilterSize>());
    }
    RealMatrix in(1, hostVecSize);
    in.row(0) = input[0];
    RealMatrix out(1, hostVecSize);
    
    mBufferedProcess.push(RealMatrixView(in));
    mBufferedProcess.processInput(
        totalWindow, get<kFFT>().hopSize(), c, [&, this](RealMatrixView in) {
          out.row(0)(mFrameOffset) = mAlgorithm.processFrame(
              in.row(0), get<kFunction>(), get<kFilterSize>(),
              get<kThreshold>(), get<kDebounce>(), get<kFrameDelta>());
          mFrameOffset += get<kFFT>().hopSize();
        });

    mFrameOffset =
        mFrameOffset < hostVecSize ? mFrameOffset : mFrameOffset - hostVecSize;

    output[0] = out.row(0);
  }

  index latency() { return static_cast<index>(get<kFFT>().hopSize()); }

  void reset()
  {    
    mBufferedProcess.reset();
    mFrameOffset = 0;
    mAlgorithm.init(get<kFFT>().winSize(), get<kFFT>().fftSize(),
                    get<kFilterSize>());
  }

private:
  OnsetSegmentation                          mAlgorithm;
  ParameterTrackChanges<index, index, index> mBufferParamsTracker;
  ParameterTrackChanges<index, index>        mParamsTracker;
  BufferedProcess                            mBufferedProcess;
  index mFrameOffset{0}; // in case kHopSize < hostVecSize
};
} // namespace onsetslice

using RTOnsetSliceClient = ClientWrapper<onsetslice::OnsetSliceClient>;

auto constexpr NRTOnsetSliceParams =
    makeNRTParams<onsetslice::OnsetSliceClient>(
        InputBufferParam("source", "Source Buffer"),
        BufferParam("indices", "Indices Buffer"));


using NRTOnsetSliceClient =
    NRTSliceAdaptor<onsetslice::OnsetSliceClient, decltype(NRTOnsetSliceParams),
                    NRTOnsetSliceParams, 1, 1>;


using NRTThreadingOnsetSliceClient = NRTThreadingAdaptor<NRTOnsetSliceClient>;

} // namespace client
} // namespace fluid
