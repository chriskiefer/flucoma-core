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
#include "../../algorithms/public/Concat.hpp"
#include <tuple>

/*
something like:
liveinput = ...
segmentTrigger = (onsets.ar(liveinput))
FluidConcat.ar(liveinput, segmentTrigger, featureSize, featureInput, historyLength(seconds), historyStart, historyWidth, overlaps)
*/
namespace fluid {
namespace client {
namespace ConcatClient {

enum ConcatParamIndex {
  kSegmentIn,
  kFeatureIn,
  kFeatureBuffer,
  kMaxHistoryLength,
  kHistoryWindowLength,
  kHistoryWindowOffset,
  kFadeTime,
  kSpeed
};


constexpr auto ConcatParams = defineParameters(
    FloatParam("segmentTrig", "segmentTrig", 0),
    FloatParam("featureTrig", "featureTrig", 0),
    BufferParam("featureBuffer", "Feature"),
    LongParam("maxHistoryLength", "Max History Length (ms)", 100, Min(100)),
    LongParam("historyWindowLength", "History Window Length (ms)", 100, Min(100)),
    LongParam("historyWindowOffset", "History Window Offset (ms)", 100, Min(100)), 
    FloatParam("fadeTime", "Fade Time (ms)", 0, Min(0)),
    FloatParam("speed", "Playback Speed", 1.0)
);

class ConcatClient : public FluidBaseClient, public AudioIn, public AudioOut
{

public:
  using ParamDescType = decltype(ConcatParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return ConcatParams; }

  ConcatClient(ParamSetViewType& p) : mParams(p)
  {
    audioChannelsIn(3);
    audioChannelsOut(1);
    FluidBaseClient::setInputLabels({"audio input"});
    FluidBaseClient::setOutputLabels({"audio out"});
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext&)
  {

    if (!input[0].data() || !output[0].data()) return;

    if (!mAlgo.isInitialised())
    { 
      mAlgo.init(sampleRate(), get<kMaxHistoryLength>()); 
      std::cout << "maxh " <<  get<kMaxHistoryLength>() << std::endl;
    }

    // buffer checking
    BufferAdaptor *featureBuf = get<kFeatureBuffer>().get();
    if (!featureBuf) return;

    //Q: Why doesn't this work with LocalBuf?
    BufferAdaptor::ReadAccess buf(featureBuf);
    if (!buf.exists() || !buf.valid())
    {
      return;
    }

    RealVector feature(buf.numFrames());
    feature = buf.samps(0, buf.numFrames(), 0);

    for (index i = 0; i < input[0].size(); i++)
    {
      output[0](i) = static_cast<T>(mAlgo.processSample(
          input[0](i),   //audio in
          input[1](i),   //segment trig
          input[2](i),   //feature trig
          feature,
          get<kHistoryWindowLength>(),
          get<kHistoryWindowOffset>(),
          get<kFadeTime>(),
          get<kSpeed>()
      ));
      // output[0](i) = static_cast<T>(input[1](i));
    }
  }

  index latency() { return 0; }

  void reset()
  {
  }

private:
  algorithm::Concat mAlgo;
};
} // namespace ampslice

using RTConcatClient = ClientWrapper<ConcatClient::ConcatClient>;

} // namespace client
} // namespace fluid
