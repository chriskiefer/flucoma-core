/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).


Chris Kiefer, 2021

*/

#pragma once

#include "../util/FluidEigenMappings.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <cmath>
#include "RingBuf.hpp"
#include <deque>

namespace fluid {
namespace algorithm {



class Concat
{
  using ivec = Eigen::Array<unsigned int, Eigen::Dynamic, 1>;
  using ArrayXd = Eigen::ArrayXd;
  using distanceFunction = std::function<double(RealVector &a, RealVector &b)>;

public:
  Concat()
  {
    euclideanDistance = [](RealVector &a, RealVector &b)
    {
      double dist = 0;
      for(size_t i = 0; i < a.size(); i++) {
        dist += pow(a[i]-b[i],2);
      }
      return sqrt(dist);
    };
    cosineDistance = [](RealVector &a, RealVector &b)
    {
      double Amag=0, Bmag=0, AB = 0;
      for(size_t i = 0; i < a.size(); i++) {
        AB += (a[i] * b[i]);
        Amag += a[i] * a[i];
        Bmag += b[i] * b[i];
      }
      double dist = AB / (sqrt(Amag) * sqrt(Bmag));
      return -dist; //invert distance so smaller is closest
    };
    distanceFunctions.push_back(euclideanDistance);
    distanceFunctions.push_back(cosineDistance);

    audioRingBuf.setSize(512);
  }

  void init(const double pSampleRate, const long maxHistoryLength)
  {
    using namespace std;
    sampleRate = pSampleRate;
    mInitialsed=1;
    mFeatureAverageCount = 0;
    mAudioRingLen = static_cast<size_t>(maxHistoryLength / 1000 * sampleRate);
    audioRingBuf.setSize(mAudioRingLen);
    cout << "Init ring buf " << mAudioRingLen << endl;
    cout << "SR " << sampleRate << endl;
    cout << "maxh " << maxHistoryLength << endl;
  }

  bool isInitialised() {return mInitialsed;}
  
  double processSample(const double audioIn, const double segmentTrig, const double featureTrig, 
  RealVector &feature, 
  const long historyWindowLength, const long historyWindowOffset, 
  const double fadeTime, const double speed)
  {
    using namespace std;
    double audioOut = 0.0;
    if (mFirst) {
      mFirst=0;
      mLastFeatureTrig = featureTrig;
      mLastSegmentTrig = segmentTrig;
    }


    //-------------- ANALYSIS
    //feature?
    if (mLastFeatureTrig <= 0 && featureTrig > 0) {
      if (mFirstFeature) {
        mFirstFeature = 0;
        mFeatureAvgBuffer = feature;
        segmentStartTS =  mSampleClock;        
      }else {
        //add feature into averaging buffer
        for(size_t i = 0;  i < mFeatureAvgBuffer.size(); i++)  mFeatureAvgBuffer[i] += feature[i];
      }
      mFeatureAverageCount++;
      // cout << mFeatureAvgBuffer << endl;
    }

    //segment?
    if (mLastSegmentTrig <= 0 && segmentTrig > 0) {

      //remove old features from db, before matching
      if (timestamps.size() > 0) {
        // cout << "oldest: " << mSampleClock - timestamps[0] << endl;
        while (mSampleClock - timestamps[0] > mAudioRingLen) {
          timestamps.pop_front();
          db.pop_front();
          // cout << "Removed old feature\n";
        }
      }

      //do averaging
      double scale = mFeatureAverageCount ? 1.0/mFeatureAverageCount : 0;
      for(size_t i = 0;  i < mFeatureAvgBuffer.size(); i++)  mFeatureAvgBuffer[i] *= scale;
      audioOut = mFeatureAvgBuffer[0];
      
      //search DB for best match 
      if (db.size() > 0) {
        // cout << "Seaching " << db.size() << " db entries\n";
        double minDist = std::numeric_limits<double>::max();
        size_t closest=0;
        for (size_t i_db=0; i_db < db.size(); i_db++) {
          double dist = euclideanDistance(mFeatureAvgBuffer, db[i_db]);
          if (dist < minDist) {
            minDist = dist;
            closest = i_db;
          }
          // cout << dist << ",";
        }
        // cout << endl;
        //segmentstartTS marks the end of the last segment in the database
        size_t matchStartOffset = segmentStartTS - timestamps[closest];
        size_t matchSize = closest == db.size()-1 ? matchStartOffset : timestamps[closest+1] - timestamps[closest];
        cout << "Match: " << closest << "/" << db.size() << "; win: " << matchSize << "; offs: "  << matchStartOffset << endl;

        auto segmentCopy = audioRingBuf.getBuffer(matchSize, matchStartOffset);
        segmentQueue.push_back(segmentCopy);
        segmentPosition.push_back(0);
        double fadeTimeSamples = (fadeTime / 1000.0) * sampleRate;
        segmentFadeLenSamples.push_back(fadeTimeSamples);
        segmentAmpMod.push_back(1.0 / fadeTimeSamples);
        segmentAmp.push_back(0);
        segmentFadeState.push_back(fadeStates::FADING_IN);
      }



      //add feature to db
      db.push_back(mFeatureAvgBuffer);
      timestamps.push_back(segmentStartTS);
      cout << "Pushing feature, db size " << db.size() << endl;



      //clear averaging buffer
      mFeatureAverageCount = 0;
      for(size_t i = 0;  i < mFeatureAvgBuffer.size(); i++)  mFeatureAvgBuffer[i] = 0;
      segmentStartTS =  mSampleClock;



    }

    mLastSegmentTrig = segmentTrig;
    mLastFeatureTrig = featureTrig;

    //-------------- SYNTHESIS
    audioRingBuf.push(audioIn);

    if (segmentQueue.size() > 0) {
      //calculate sample with linear interpolation
      audioOut = 0;
      //for each segment in the queue
      for(size_t segIdx=0; segIdx < segmentQueue.size(); segIdx++) {
        double seqmentSample=0.0;
        size_t sampleIndex = static_cast<size_t>(segmentPosition[segIdx]);
        if (speed >= 0) {
          double i0 = segmentPosition[segIdx] - sampleIndex;
          double i1 = 1.0 - i0;
          size_t nextSampleIndex = sampleIndex + 1;
          if (nextSampleIndex == segmentQueue[segIdx].size()) nextSampleIndex = 0;
          seqmentSample = (i0 * segmentQueue[segIdx][sampleIndex]) + (i1 * segmentQueue[segIdx][nextSampleIndex]);
        }else{
          size_t firstSampleIndex = sampleIndex-1;
          if (firstSampleIndex < 0) {
            firstSampleIndex = segmentQueue[segIdx].size()-1;
          }
          double i1 = segmentPosition[segIdx] - sampleIndex;
          double i0 = 1.0 - i0;
          seqmentSample = (i0 * segmentQueue[segIdx][firstSampleIndex]) + (i1 * segmentQueue[segIdx][sampleIndex]);
        }

        //cross-fades
        seqmentSample *= sqrt(segmentAmp[segIdx]);

        segmentAmp[segIdx] += segmentAmpMod[segIdx];

        switch(segmentFadeState[segIdx]) {
          case fadeStates::FADING_IN:
            {
              if (segmentAmp[segIdx] >= 1.0) {
                segmentAmpMod[segIdx] = 0;
                segmentFadeState[segIdx] = fadeStates::MID;
                cout << "mid\n";
              }
            }
          break;
          case fadeStates::MID:
            {
              //when to trigger fade out?
              if (speed >= 0) {
                if (segmentQueue[segIdx].size() - segmentPosition[segIdx] < segmentFadeLenSamples[segIdx] ) {
                  segmentFadeState[segIdx] = fadeStates::FADING_OUT;
                }

              }else{
                if (segmentPosition[segIdx] < segmentFadeLenSamples[segIdx] ) {
                  segmentFadeState[segIdx] = fadeStates::FADING_OUT;
                }
              }
              if (segmentFadeState[segIdx] == fadeStates::FADING_OUT) {
                cout << "out\n";
                segmentAmpMod[segIdx] = -1.0 / segmentFadeLenSamples[segIdx];
                //only segment? -> add copy to the queue
                if (segmentQueue.size() == 1) {
                }
              };
            }
          break;

          case fadeStates::FADING_OUT:
            if (segmentAmp[segIdx] <= 0.0) {
              //remove segment
              segmentQueue.pop_front();
              segmentPosition.pop_front();
              segmentAmp.pop_front();
              segmentAmpMod.pop_front();
              segmentFadeLenSamples.pop_front();
              segmentFadeState.pop_front();
            }

          break;
        }


        

        //adjust position
        segmentPosition[segIdx]+= speed;
        if (segmentPosition[segIdx]>=segmentQueue[segIdx].size()) {
          segmentPosition[segIdx] -= segmentQueue[segIdx].size();
        }else if (segmentPosition[segIdx] < 0){
          segmentPosition[segIdx] += segmentQueue[segIdx].size();
        }
        
        audioOut += seqmentSample;
      }

    }else{
      audioOut = 0;
    }
    

    
    mSampleClock++;
    return audioOut;
  }

  index getLatency() { return 0; }



private:
  enum fadeStates {FADING_IN, MID, FADING_OUT};
  RingBuf<double> audioRingBuf;
  bool mInitialsed = 0;
  bool mFirstFeature=1;
  bool mFirstSegment=1;
  bool mFirst=1;
  RealVector mFeatureAvgBuffer;
  size_t mFeatureAverageCount;
  double mLastSegmentTrig;
  double mLastFeatureTrig;
  std::deque<RealVector> db;
  std::deque<size_t> timestamps;
  size_t segmentStartTS=0;
  size_t mSampleClock=0;
  distanceFunction euclideanDistance;
  distanceFunction cosineDistance;
  std::vector<distanceFunction> distanceFunctions;
  double sampleRate=44100;
  size_t mAudioRingLen;
  std::deque<RingBuf<double>::winBufType> segmentQueue;
  std::deque<double> segmentPosition;
  std::deque<double> segmentAmpMod;
  std::deque<double> segmentAmp;
  std::deque<double> segmentFadeLenSamples;
  std::deque<fadeStates> segmentFadeState;
  

};



} // namespace algorithm
} // namespace fluid


//TODO NEXT:  compile various seqment queues into single queue + data structure
// then (1) add in segment copy if current segment finishes on its own (2) fade out top segment when adding a new one
// then (3) logic for moving history matching window