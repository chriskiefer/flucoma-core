#pragma once

#include "BufferedProcess.hpp"
#include "../common/ParameterTypes.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTrackChanges.hpp"
#include "../../algorithms/public/NMF.hpp"
#include "../../algorithms/public/RatioMask.hpp"

namespace fluid {
namespace client {

class NMFFilter : public FluidBaseClient, public AudioIn, public AudioOut
{

  enum NMFFilterIndex{kFilterbuf,kMaxRank,kIterations,kFFT,kMaxFFTSize};

public:

  FLUID_DECLARE_PARAMS(
    InputBufferParam("bases", "Bases Buffer"),
    LongParam<Fixed<true>>("maxComponents","Maximum Number of Components",20,Min(1)),
    LongParam("iterations", "Number of Iterations", 10, Min(1)),
    FFTParam<kMaxFFTSize>("fftSettings","FFT Settings",1024, -1,-1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4), PowerOfTwo{})
  );

  NMFFilter(ParamSetViewType& p) : mParams(p), mSTFTProcessor(get<kMaxFFTSize>(),1,get<kMaxRank>())
  {
    audioChannelsIn(1);
    audioChannelsOut(get<kMaxRank>());
  }

  size_t latency() { return get<kFFT>().winSize(); }

  template <typename T>
  void process(std::vector<HostVector<T>> &input, std::vector<HostVector<T>> &output, FluidContext& c,
               bool reset = false) 
  {
    if(!input[0].data()) return;
    assert(audioChannelsOut() && "No control channels");
    assert(output.size() >= audioChannelsOut() && "Too few output channels");

    if (get<kFilterbuf>().get()) {

      auto filterBuffer = BufferAdaptor::ReadAccess(get<kFilterbuf>().get());
      auto& fftParams = get<kFFT>();

      if (!filterBuffer.valid()) {
        return ;
      }

      size_t rank  = std::min<size_t>(filterBuffer.numChans(),get<kMaxRank>());

      if (filterBuffer.numFrames() != fftParams.frameSize())
      {
        return;
      }

      if(mTrackValues.changed(rank, fftParams.frameSize()))
      {
        tmpFilt.resize(rank,fftParams.frameSize());
        tmpMagnitude.resize(1,fftParams.frameSize());
        tmpOut.resize(rank);
        tmpEstimate.resize(1,fftParams.frameSize());
        tmpSource.resize(1,fftParams.frameSize());
        mNMF.reset(new algorithm::NMF(rank, get<kIterations>()));
      }

      for (size_t i = 0; i < tmpFilt.rows(); ++i)
        tmpFilt.row(i) = filterBuffer.samps(i);

//      controlTrigger(false);
      mSTFTProcessor.process(mParams, input, output, c, reset,
        [&](ComplexMatrixView in,ComplexMatrixView out)
        {
          algorithm::STFT::magnitude(in, tmpMagnitude);
          mNMF->processFrame(tmpMagnitude.row(0), tmpFilt, tmpOut,get<kIterations>(),tmpEstimate.row(0));
          auto mask = algorithm::RatioMask{tmpEstimate,1};
          for(size_t i = 0; i < rank; ++i)
          {
            algorithm::NMF::estimate(tmpFilt,RealMatrixView(tmpOut),i,tmpSource);
            mask.process(in,RealMatrixView{tmpSource},ComplexMatrixView{out.row(i)});
          }
        });
    }
  }

private:
  ParameterTrackChanges<size_t,size_t> mTrackValues;
  STFTBufferedProcess<ParamSetViewType, kFFT,true> mSTFTProcessor;
  std::unique_ptr<algorithm::NMF> mNMF;

  RealMatrix a;

  RealMatrix tmpFilt;
  RealMatrix tmpMagnitude;
  RealVector tmpOut;
  RealMatrix tmpEstimate;
  RealMatrix tmpSource;

  size_t mNBins{0};
  size_t mRank{0};
};

using RTNMFFilterClient = ClientWrapper<NMFFilter>; 

} // namespace client
} // namespace fluid
