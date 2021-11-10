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

#include "../util/FluidEigenMappings.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <cmath>

#include <unordered_map>

namespace fluid {
namespace algorithm {

template <typename T,size_t N>
class RingBuf {
using ringBufType = Eigen::Array<T,N,1>;
using winBufType = Eigen::Array<T,Eigen::Dynamic,1>;
public:
    RingBuf() {
        // buf.resize(16);
        buf.fill(0);
        currBuf.resize(512);
    }
    // RingBuf(size_t W) {
    //     buf.resize(W);
    //     buf.fill(0);
    // }
    // void setSize(size_t N) {
    //     buf.resize(N);
    //     buf.fill(0);
    //     idx=0;
    // }

    void push(T x) {
        buf[idx] = x;
        idx++;
        if (idx==buf.size()) {
            idx=0;
        }
    }
    
    size_t size() {return buf.size();}
    
    winBufType& getBuffer(unsigned int winSize) {
        if (winSize != buf.size())
          currBuf.resize(winSize);
        int targidx=0;
        if (idx > winSize) {
            for(int i=idx-winSize; i < idx; i++, targidx++) {
                currBuf[targidx] = buf[i];
            }
        }else{
            //first chunk
            for(int i=buf.size()-(winSize-idx); i < buf.size(); i++, targidx++) {
                currBuf[targidx] = buf[i];
            }
            //second chunk
            for(int i=0; i < idx; i++, targidx++) {
                currBuf[targidx] = buf[i];
            }
        }
        return currBuf;
    }
    
private:
    ringBufType buf;
    size_t idx=0;
    winBufType currBuf;

};



class ShannonEntropy
{
  using ivec = Eigen::Array<unsigned int, Eigen::Dynamic, 1>;
  using ArrayXd = Eigen::ArrayXd;
  using histoMap= std::unordered_map<int, unsigned int>;

public:
  ShannonEntropy()
  {
  }

  void init()
  {
    using namespace std;
    mInitialsed=1;
  }

  bool isInitialised() {return mInitialsed;}

  double processSample(const double in, const double symbolCount, const long hopSize)
  {
    using namespace std;
    //symbolise
    //normalise 0-1
    double inNorm = (in + 1.0) / 2.0;
    //clamp
    inNorm = inNorm < 0 ? 0 : inNorm;
    inNorm = inNorm > 1 ? 1 : inNorm;
    //quantise
    unsigned int inSym = static_cast<unsigned int>(inNorm * symbolCount);
    ringBuf.push(inSym);
    hopCounter++;
    if (hopCounter >= hopSize) {
      hopCounter=0;
      auto window = ringBuf.getBuffer(256);
      entropy = calcFromWindow(window);
    }
    return entropy;
  }

  index getLatency() { return 0; }
//   bool  initialized() { return mInitialized; }


  histoMap calcDistribution(const ivec &seq) {
      histoMap histo;
      for (auto v: seq) {
          histoMap::iterator it = histo.find(v);
          if (it == histo.end()) {
              histo.insert(std::make_pair<int, unsigned int>(std::move(v), 1));
          }else{
              it->second = it->second + 1;
          }
      }
      return histo;
  }

  double calcProbability(const histoMap &histo, const ivec &seq) {
      double scale = 1.0 / seq.size();
      double H=0;
      for(auto v: histo) {
          double prob = v.second * scale;
          H = H - (prob * log2(prob));
      }
      return H;
  }

  double calcFromWindow(const ivec &seq) {
      histoMap histo = calcDistribution(seq);
      double prob = calcProbability(histo, seq);
      return prob;
  }

private:
  RingBuf<unsigned int,512> ringBuf;
  size_t hopCounter=0;
  double entropy=0;
  bool mInitialsed = 0;
};


class ETC
{

  using ArrayXd = Eigen::ArrayXd;

public:
  ETC()
  {
  }

  void init()
  {
    using namespace std;

  }

  double processSample(const double in)
  {
    using namespace std;
    // assert(mInitialized);

    return tanh(in);
  }

  index getLatency() { return 0; }
//   bool  initialized() { return mInitialized; }


private:
};

} // namespace algorithm
} // namespace fluid
