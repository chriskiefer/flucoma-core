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
#include "../common/ParameterTrackChanges.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/EnvelopeGate.hpp"
#include <tuple>

namespace fluid {
namespace client {

enum AmpGateParamIndex {
  kRampUpTime,
  kRampDownTime,
  kOnThreshold,
  kOffThreshold,
  kMinEventDuration,
  kMinSilenceDuration,
  kMinTimeAboveThreshold,
  kMinTimeBelowThreshold,
  kUpwardLookupTime,
  kDownwardLookupTime,
  kHiPassFreq,
  kMaxSize
};

auto constexpr AmpGateParams = defineParameters(
    LongParam("RampUp", "Ramp Up Length", 10, Min(1)),
    LongParam("RampDown", "Ramp Down Length", 10, Min(1)),
    FloatParam("OnThreshold", "On Threshold", -90, Min(-144), Max(144)),
    FloatParam("OffThreshold", "Off Threshold", -90, Min(-144), Max(144)),
    LongParam("minSliceLength", "Minimum Length of Slice", 1, Min(1)),
    LongParam("minSilenceLength", "Minimum Length of Silence", 1, Min(1)),
    LongParam("minLengthAbove", "Required Minimum Length Above Threshold", 1,
              Min(1)),
    LongParam("minLengthBelow", "Required Minimum Length Below Threshold", 1,
              Min(1)),
    LongParam("lookBack", "Backward Lookup Length", 0, Min(0)),
    LongParam("lookAhead", "Forward Lookup Length", 0, Min(0)),
    FloatParam("highPassFreq", "High-Pass Filter Cutoff", 85, Min(1)),
    LongParam<Fixed<true>>("maxSize", "Maximum Total Latency", 88200, Min(1)));

template <typename T>
class AmpGateClient
    : public FluidBaseClient<decltype(AmpGateParams), AmpGateParams>,
      public AudioIn,
      public AudioOut
{
  using HostVector = FluidTensorView<T, 1>;

public:
  AmpGateClient(ParamSetViewType& p) : FluidBaseClient(p)
  {
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::audioChannelsOut(1);
  }

  void process(std::vector<HostVector>& input, std::vector<HostVector>& output, FluidContext&)
  {

    if (!input[0].data() || !output[0].data()) return;

    double hiPassFreq = std::min(get<kHiPassFreq>() / sampleRate(), 0.5);

    if (mTrackValues.changed(get<kMinTimeAboveThreshold>(),
                             get<kUpwardLookupTime>(),
                             get<kMinTimeBelowThreshold>(),
                             get<kDownwardLookupTime>(), sampleRate()) ||
        !mAlgorithm.initialized())
    {
      mAlgorithm.init(hiPassFreq, get<kRampUpTime>(), get<kRampDownTime>(),
                      get<kOnThreshold>(), get<kMinTimeAboveThreshold>(),
                      get<kMinEventDuration>(), get<kUpwardLookupTime>(),
                      get<kOffThreshold>(), get<kMinTimeBelowThreshold>(),
                      get<kMinSilenceDuration>(), get<kDownwardLookupTime>());
    }
    else
    {
      mAlgorithm.updateParams(hiPassFreq, get<kRampUpTime>(),
                              get<kRampDownTime>(), get<kOnThreshold>(),
                              get<kMinEventDuration>(), get<kOffThreshold>(),
                              get<kMinSilenceDuration>());
    }

    for (index i = 0; i < input[0].size(); i++)
    { output[0](i) = static_cast<T>(mAlgorithm.processSample(input[0](i))); }
  }

  void reset() {}

  index latency()
  {
    return std::max(
        get<kMinTimeAboveThreshold>() + get<kUpwardLookupTime>(),
        std::max(get<kMinTimeBelowThreshold>(), get<kDownwardLookupTime>()));
  }

private:
  ParameterTrackChanges<index, index, index, index, double> mTrackValues;
  algorithm::EnvelopeGate mAlgorithm{get<kMaxSize>()};
};

template <typename HostMatrix, typename HostVectorView>
struct NRTAmpGate
{
  template <typename Client, typename InputList, typename OutputList>
  static Result process(Client& client, InputList& inputBuffers,
                        OutputList& outputBuffers, index nFrames, index nChans,
                        FluidContext& c)
  {
    assert(inputBuffers.size() == 1);
    assert(outputBuffers.size() == 1);
    index padding = client.latency();
    using HostMatrixView = FluidTensorView<typename HostMatrix::type, 2>;
    HostMatrix monoSource(1, nFrames + padding);

    BufferAdaptor::ReadAccess src(inputBuffers[0].buffer);
    // Make a mono sum;
    for (index i = 0; i < nChans; ++i)
      monoSource.row(0)(Slice(0, nFrames))
          .apply(src.samps(inputBuffers[0].startFrame, nFrames, i),
                 [](float& x, float y) { x += y; });

    HostMatrix switchPoints(2, nFrames);
    HostMatrix binaryOut(1, nFrames + padding);

    std::vector<HostVectorView> input{monoSource.row(0)};
    std::vector<HostVectorView> output{binaryOut.row(0)};

    // convert binary to spikes
    client.process(input, output, c);

    // add onset at start if needed
    if (output[0](padding) == 1) { switchPoints(0, 0) = 1; }
    for (index i = 1; i < nFrames - 1; i++)
    {
      if (output[0](padding + i) == 1 && output[0](padding + i - 1) == 0)
        switchPoints(0, i) = 1;
      else
        switchPoints(0, i) = 0;
      if (output[0](padding + i) == 0 && output[0](padding + i - 1) == 1)
        switchPoints(1, i) = 1;
      else
        switchPoints(1, i) = 0;
    }
    // add offset at end if needed
    if (output[0](nFrames + padding - 1) == 1)
    { switchPoints(1, nFrames - 1) = 1; }

    return impl::spikesToTimes(HostMatrixView{switchPoints}, outputBuffers[0],
                               1, inputBuffers[0].startFrame, nFrames,
                               src.sampleRate());
  }
};

auto constexpr NRTAmpGateParams =
    makeNRTParams<AmpGateClient>(InputBufferParam("source", "Source Buffer"),
                                 BufferParam("indices", "Indices Buffer"));

template <typename T>
using NRTAmpGateClient =
    impl::NRTClientWrapper<NRTAmpGate, AmpGateClient<T>,
                           decltype(NRTAmpGateParams), NRTAmpGateParams, 1, 1>;

template <typename T>
using NRTThreadedAmpGateClient = NRTThreadingAdaptor<NRTAmpGateClient<T>>;

} // namespace client
} // namespace fluid
