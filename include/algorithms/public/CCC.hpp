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
#include "RingBuf.hpp"

#include <unordered_map>

namespace fluid {
namespace algorithm {



class ShannonEntropy
{
  using ArrayXL = Eigen::Array<int64_t, Eigen::Dynamic, 1>; 

public:
  using histoMap= std::unordered_map<int64_t, unsigned int>;

  ShannonEntropy()
  {
  }

  void init(const double sampleRate, double maxWindowSize)
  {
    mSampleRate = sampleRate;
    mMaxWindowSize = static_cast<size_t>(maxWindowSize / 1000.0 * mSampleRate);
    ringBuf.setSize(mMaxWindowSize);
    mInitialsed=1;
  }

  bool isInitialised() {return mInitialsed;}

  double processSample(const double in, const long symbolCount, const double windowSize, const double hopSize)
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
    size_t hopSizeInSamples = static_cast<size_t>(hopSize / 1000.0 * mSampleRate);
    if (hopCounter >= min(hopSizeInSamples,mMaxWindowSize)) {
      hopCounter=0;
      size_t windowSizeInSamples = static_cast<size_t>(windowSize / 1000.0 * mSampleRate);
      auto window = ringBuf.getBuffer(min(windowSizeInSamples, mMaxWindowSize));
      entropy = calc(window);
    }
    return entropy;
  }

  index getLatency() { return 0; }

  histoMap calcDistribution(const ArrayXL &seq) {
      histoMap histo;
      for (const uint64_t &v: seq) {
          histoMap::iterator it = histo.find(v);
          if (it == histo.end()) {
              histo.insert(std::make_pair(v, 1));
          }else{
              it->second = it->second + 1;
          }
      }
      return histo;
  }
  
  double calcProbability(const histoMap &histo, const ArrayXL &seq) {
      double scale = 1.0 / seq.size();
      double H=0;
      for(auto v: histo) {
          double prob = v.second * scale;
          H = H - (prob * log2(prob));
      }
      return H;
  }
  
  double calc(const ArrayXL &seq) {
      histoMap histo = calcDistribution(seq);
      double prob = calcProbability(histo, seq);
      return prob;
  }

private:
  RingBuf<int64_t> ringBuf;
  size_t hopCounter=0;
  double entropy=0;
  bool mInitialsed = 0;
  size_t mMaxWindowSize;
  double mSampleRate;
};


class ETC
{

using ArrayXL = Eigen::Array<int64_t, Eigen::Dynamic, 1>; 
// using pairFreqTable = std::unordered_map<__int128, unsigned int> ;
using pairFreqTable = std::map<__int128, unsigned int> ;

public:

  union pair {
      struct {
          uint64_t i1;
          uint64_t i2;
      } __attribute__((packed));
      __int128 i128;

      bool operator==(const pair& other)
      {
          return i128 == other.i128;
      }
  };


  ETC()
  {
  }

  pair makeETCPair (uint64_t a, uint64_t b) {
    pair p; p.i1 = a; p.i2 = b; return p;
  };

  void init(const double sampleRate, double maxWindowSize)
  {
    mSampleRate = sampleRate;
    mMaxWindowSize = static_cast<size_t>(maxWindowSize / 1000.0 * mSampleRate);
    ringBuf.setSize(mMaxWindowSize);
    mInitialsed=1;
  }

  bool isInitialised() {return mInitialsed;}

  pair findHFPair(const ArrayXL &seq) {
      /*
        this implementation works very slightly differently from the matlab original where more that one pair wins the highest frequency
        - the matlab version depends on arbitrary behaviour of max(x)
        [m,indx]=max(Count_Array(:));
        i.e. chooses winner based on first position in 2d frequency matrix
        whereas this version chooses the first winner in the order of the array, which is slightly faster
        tests show this occasionally makes very minor differences in the results
        */
      ETC::pairFreqTable histo;
      pair winner;
      unsigned int highScore=0;

      unsigned int seqPos = 0;
      while (seqPos < seq.size()-1) {
          pair currPair = makeETCPair(seq[seqPos], seq[seqPos+1]);
          pairFreqTable::iterator it = histo.find(currPair.i128);
          unsigned int score;
          if (it == histo.end()) {
              histo.insert(std::make_pair(currPair.i128, 1));
              score=1;
          }else{
              score = it->second + 1;
              it->second = score;
          }
          if (score > highScore) {
              highScore = score;
              winner = currPair;
          }
          if (currPair.i1 == currPair.i2) {
              if (seqPos < seq.size()-2) {
                  if (seq[seqPos+2] == seq[seqPos]) {
                      seqPos++;
                  }
              }
          }
          seqPos++;
      }
      return winner;
  }
  

  auto substitute(const ArrayXL &seq, pair p) {
      int64_t replacementSymbol = seq.maxCoeff() + 1;
      ArrayXL newSeq(seq.size());
      size_t src=0, dest=0;
      size_t replaceCount=0;
      while(src < seq.size()) {
          if (src < seq.size()-1) {
              if (seq[src] == p.i1 && seq[src+1] == p.i2) {
                  newSeq[dest] = replacementSymbol;
                  src++;
                  replaceCount++;
              } else {
                  newSeq[dest] = seq[src];
              }
          }else{
              newSeq[dest] = seq[src];
          }
          src++;
          dest++;
      }
      ArrayXL finalSeq = newSeq.head(dest);
      return std::make_tuple(finalSeq, replaceCount, replacementSymbol);
  }

  double calc(const ArrayXL &seq) {
      double N = 0; //ETC measure
      if (seq.size() > 1) {
          ShannonEntropy::histoMap histo = shannon.calcDistribution(seq);
          double Hnew = shannon.calcProbability(histo, seq);

          ArrayXL newSeq = seq;
          
          while(Hnew >1e-6 && newSeq.size() > 1) {
              pair hfPair = ETC::findHFPair(newSeq);
              auto [newSeqRepl, replaceCount, replaceSym] = substitute(newSeq, hfPair);
              //reduce counts of replacement pair
              ShannonEntropy::histoMap::iterator it = histo.find(hfPair.i1);
              it->second -= replaceCount;
              if (it->second == 0) {
                  //remove from the histo
                  histo.erase(it);
              }
              it = histo.find(hfPair.i2);
              it->second -= replaceCount;
              if (it->second == 0) {
                  //remove from the histo
                  histo.erase(it);
              }
              //add the new symbol into the histogram
              auto histoEntry = std::make_pair(replaceSym, replaceCount);
              histo.insert(histoEntry);
              
              
              Hnew = shannon.calcProbability(histo, newSeqRepl);
              newSeq = newSeqRepl;
              N++;
          }
          N /= (seq.size() -1);
      }
      return N;
  }
  
  double calcJoint(const ArrayXL& seq1, const ArrayXL& seq2) {
      
      ArrayXL combSeq(seq1.size());
      for (size_t i=0; i < seq1.size(); i++) {
          combSeq[i] =(uint64_t)( seq1[i] | (seq2[i] << 32));
      }
      return ETC::calc(combSeq);
  }


  double processSample(const double in, const long symbolCount, const double windowSize, const double hopSize)
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
    size_t hopSizeInSamples = static_cast<size_t>(hopSize / 1000.0 * mSampleRate);
    if (hopCounter >= min(hopSizeInSamples,mMaxWindowSize)) {
      hopCounter=0;
      size_t windowSizeInSamples = static_cast<size_t>(windowSize / 1000.0 * mSampleRate);
      auto window = ringBuf.getBuffer(min(windowSizeInSamples, mMaxWindowSize));
      etc = calc(window);
    }
    return etc;
  }

  index getLatency() { return 0; }


private:
  RingBuf<int64_t> ringBuf;
  size_t hopCounter=0;
  bool mInitialsed = 0;
  size_t mMaxWindowSize;
  double mSampleRate;
  ShannonEntropy shannon;
  double etc=0.0;
};

} // namespace algorithm
} // namespace fluid
