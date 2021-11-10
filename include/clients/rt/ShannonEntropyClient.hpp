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
#include "../../algorithms/public/CCC.hpp"
#include <tuple>

namespace fluid {
namespace client {
namespace ShannonEntropy {

enum ShannonEntropyParamIndex {
  kSymCount,
  kWinSize,
  kMaxWinSize,
  kHopSize
};

constexpr auto ShannonEntropyParams = defineParameters(
    LongParam("symbolCount", "Symbol Count", 1, Min(1)),
    LongParam("winSize", "Window Size", 512, Min(4)),
    LongParam("maxWinSize", "Maximum Window Size", 512, Min(4)),
    LongParam("hopSize", "Hop Size", 256, Min(4))
);

class ShannonEntropyClient : public FluidBaseClient, public AudioIn, public AudioOut
{

public:
  using ParamDescType = decltype(ShannonEntropyParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return ShannonEntropyParams; }

  ShannonEntropyClient(ParamSetViewType& p) : mParams(p)
  {
    audioChannelsIn(1);
    audioChannelsOut(1);
    FluidBaseClient::setInputLabels({"audio input"});
    FluidBaseClient::setOutputLabels({"Shannon Entropy"});
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext&)
  {

    if (!input[0].data() || !output[0].data()) return;

    if (!mAlgo.isInitialised())
    { 
      mAlgo.init(); 
    }

    for (index i = 0; i < input[0].size(); i++)
    {
      output[0](i) = static_cast<T>(mAlgo.processSample(
          input[0](i), 
          get<kSymCount>(),  
          get<kHopSize>()
      ));
    }
  }

  index latency() { return 0; }

  void reset()
  {
  }

private:
  algorithm::ShannonEntropy mAlgo;
};
} // namespace ampslice

using RTShannonEntropyClient = ClientWrapper<ShannonEntropy::ShannonEntropyClient>;

} // namespace client
} // namespace fluid
