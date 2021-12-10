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
  kControlSegmentIn,
  kControlFeatureIn,
  kSourceSegmentIn,
  kSourceFeatureIn,
  kControlFeatureBuffer,
  kSourceFeatureBuffer,
  kMaxHistoryLength,
  kHistoryWindowLength,
  kHistoryWindowOffset,
  kFadeTime,
  kSpeed, 
  kAlgo,
  kRandomness
};


constexpr auto ConcatParams = defineParameters(
    FloatParam("controlSegmentTrig", "controlSegmentTrig", 0),
    FloatParam("controlFeatureTrig", "controlFeatureTrig", 0),
    FloatParam("sourceSegmentTrig", "sourceSegmentTrig", 0),
    FloatParam("sourceFeatureTrig", "sourceFeatureTrig", 0),
    BufferParam("controlFeatureBuffer", "Control Feature"),
    BufferParam("sourceFeatureBuffer", "Source Feature"),
    LongParam("maxHistoryLength", "Max History Length (ms)", 100, Min(100)),
    LongParam("historyWindowLength", "History Window Length (ms)", 100, Min(100)),
    LongParam("historyWindowOffset", "History Window Offset (ms)", 100, Min(100)), 
    FloatParam("fadeTime", "Fade Time (ms)", 0, Min(0)),
    FloatParam("speed", "Playback Speed", 1.0),
    LongParam("algo", "Matching Algorithm", 0, Min(0), Max(1)),
    FloatParam("randomness", "Randomness", 0, Min(0), Max(1))
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
    audioChannelsIn(5);
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
    BufferAdaptor *sourceFeatureBuf = get<kSourceFeatureBuffer>().get();
    if (!sourceFeatureBuf) return;
    BufferAdaptor *controlFeatureBuf = get<kControlFeatureBuffer>().get();
    if (!controlFeatureBuf) return;

    //Q: Why doesn't this work with LocalBuf?
    BufferAdaptor::ReadAccess sourceBuf(sourceFeatureBuf);
    if (!sourceBuf.exists() || !sourceBuf.valid())
    {
      return;
    }
    BufferAdaptor::ReadAccess controlBuf(controlFeatureBuf);
    if (!controlBuf.exists() || !controlBuf.valid())
    {
      return;
    }

    RealVector sourceFeatureVect(sourceBuf.numFrames());
    sourceFeatureVect = sourceBuf.samps(0, sourceBuf.numFrames(), 0);
    RealVector controlFeatureVect(controlBuf.numFrames());
    controlFeatureVect = controlBuf.samps(0, controlBuf.numFrames(), 0);

    for (index i = 0; i < input[0].size(); i++)
    {
      output[0](i) = static_cast<T>(mAlgo.processSample(
          input[0](i),   //audio source in
          input[1](i),   //control segment trig
          input[2](i),   //control feature trig
          input[3](i),   //source segment trig
          input[4](i),   //source feature trig
          controlFeatureVect,
          sourceFeatureVect,
          get<kHistoryWindowLength>(),
          get<kHistoryWindowOffset>(),
          get<kFadeTime>(),
          get<kSpeed>(),
          get<kAlgo>(),
          get<kRandomness>()
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
