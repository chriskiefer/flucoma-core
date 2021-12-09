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
#include <random>


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
    //next: add LSH distance
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
  
  double processSample(const double audioIn, double segmentTrig, const double featureTrig, 
    RealVector &feature, 
    const long historyWindowLength, const long historyWindowOffset, 
    const double fadeTime, const double speed, const unsigned int matchingAlgo,
    const double randomness)
  {
    using namespace std;
    double audioOut = 0.0;
    if (mFirst) {
      mFirst=0;
      mLastFeatureTrig = featureTrig;
      mLastSegmentTrig = segmentTrig;
    }
    //auto trig if starting to get too long
    if (mSampleClock - segmentStartTS > (static_cast<size_t>(historyWindowLength / 1000) * sampleRate * 0.5)) {
      segmentTrig=1;
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
        size_t selectedSegment=0;
        size_t offsetInSamples = (static_cast<size_t>(historyWindowOffset / 1000) * sampleRate);
        size_t searchWindowStart = segmentStartTS > offsetInSamples ? segmentStartTS - offsetInSamples : 0;
        size_t searchWindowEnd = searchWindowStart + (static_cast<size_t>(historyWindowLength / 1000) * sampleRate);
        // cout << "searching from " << searchWindowStart << " to " << searchWindowEnd << endl;
        //keep list of distances for random selection
        std::vector< std::pair<double, size_t> > distances;
        for (size_t i_db=0; i_db < db.size(); i_db++) {
          if (timestamps[i_db] >= searchWindowStart && timestamps[i_db] < searchWindowEnd) {
            double dist = distanceFunctions[matchingAlgo](mFeatureAvgBuffer, db[i_db]);
            if (dist < minDist) {
              minDist = dist;
              selectedSegment = i_db;
            }
            if (randomness > 0.0) {
              distances.push_back(std::make_pair(dist, i_db));
            }
          }
          // cout << dist << ",";
        }
        if (randomness > 0.0) {
          std::sort(distances.begin(), distances.end());
          size_t maxIndex = static_cast<size_t>(round((distances.size()-1) * randomness));
          std::uniform_int_distribution<int> distribution(0,maxIndex);
          selectedSegment = distribution(mtrand);
          cout << "r: " << selectedSegment << endl;
        }
        // cout << endl;
        //segmentstartTS marks the end of the last segment in the database
        size_t matchStartOffset = segmentStartTS - timestamps[selectedSegment];
        size_t matchSize = selectedSegment == db.size()-1 ? matchStartOffset : timestamps[selectedSegment+1] - timestamps[selectedSegment];
        // cout << "Match: " << closest << "/" << db.size() << "; win: " << matchSize << "; offs: "  << matchStartOffset << endl;

        //fade out the other segments in the queue
        for(size_t segIdx = 0; segIdx < mSegmentQ.size(); segIdx++) {
          mSegmentQ[segIdx].startFadeOut();
        }

        soundSegment newSegment;
        newSegment.segmentAudio =  audioRingBuf.getBuffer(matchSize, matchStartOffset);
        newSegment.position = speed > 0 ? 0 : newSegment.segmentAudio.size() - 1;
        newSegment.fadeLenSamples = (fadeTime / 1000.0) * sampleRate;
        newSegment.speed = speed;
        newSegment.startFadeIn();
        mSegmentQ.push_back(newSegment);
        // cout << "new seg, q size: " << mSegmentQ.size() << ", amp: " << newSegment.amp << endl;



      }



      //add feature to db
      db.push_back(mFeatureAvgBuffer);
      timestamps.push_back(segmentStartTS);
      // cout << "Pushing feature, db size " << db.size() << endl;



      //clear averaging buffer
      mFeatureAverageCount = 0;
      for(size_t i = 0;  i < mFeatureAvgBuffer.size(); i++)  mFeatureAvgBuffer[i] = 0;
      segmentStartTS =  mSampleClock;



    }

    mLastSegmentTrig = segmentTrig;
    mLastFeatureTrig = featureTrig;

    //-------------- SYNTHESIS
    audioRingBuf.push(audioIn);

    if (mSegmentQ.size() > 0) {
      audioOut = 0;
      //for each segment in the queue
      for(size_t segIdx=0; segIdx < mSegmentQ.size(); segIdx++) {
        //calculate sample with linear interpolation
        double seqmentSample=0.0;
        size_t sampleIndex = static_cast<size_t>(mSegmentQ[segIdx].position);
        if (mSegmentQ[segIdx].speed >= 0) {
          double i0 = mSegmentQ[segIdx].position - sampleIndex;
          double i1 = 1.0 - i0;
          size_t nextSampleIndex = sampleIndex + 1;
          if (nextSampleIndex == mSegmentQ[segIdx].segmentAudio.size()) nextSampleIndex = 0;
          seqmentSample = (i0 * mSegmentQ[segIdx].segmentAudio[sampleIndex]) + (i1 * mSegmentQ[segIdx].segmentAudio[nextSampleIndex]);
        }else{
          size_t firstSampleIndex = sampleIndex-1;
          if (firstSampleIndex == 0) {
            firstSampleIndex = mSegmentQ[segIdx].segmentAudio.size()-1;
          }
          double i1 = mSegmentQ[segIdx].position - sampleIndex;
          double i0 = 1.0 - i1;
          seqmentSample = (i0 * mSegmentQ[segIdx].segmentAudio[firstSampleIndex]) + (i1 * mSegmentQ[segIdx].segmentAudio[sampleIndex]);
        }

        //equal power cross-fades
        seqmentSample *= sqrt(mSegmentQ[segIdx].amp);

        mSegmentQ[segIdx].amp += mSegmentQ[segIdx].ampMod;

        switch(mSegmentQ[segIdx].fadeState) {
          case fadeStates::FADING_IN:
            {
              if (mSegmentQ[segIdx].amp >= 1.0) {
                mSegmentQ[segIdx].ampMod = 0;
                mSegmentQ[segIdx].fadeState = fadeStates::MID;
                // cout << "mid\n";
              }
            }
          break;
          case fadeStates::MID:
            {
              //when to trigger fade out?
              if (mSegmentQ[segIdx].speed >= 0) {
                if (mSegmentQ[segIdx].segmentAudio.size() - mSegmentQ[segIdx].position < mSegmentQ[segIdx].fadeLenSamples ) {
                  mSegmentQ[segIdx].startFadeOut();
                }

              }else{
                if (mSegmentQ[segIdx].position < mSegmentQ[segIdx].fadeLenSamples ) {
                  mSegmentQ[segIdx].startFadeOut();
                }
              }
              if (mSegmentQ[segIdx].fadeState == fadeStates::FADING_OUT) {
                //only segment? -> add copy to the queue
                if (mSegmentQ.size() == 1) {
                  // cout << "Cloning seg" << endl;

                  soundSegment segCopy = mSegmentQ[segIdx];
                  segCopy.position=0;
                  segCopy.startFadeIn();
                  mSegmentQ.push_back(segCopy);  
                }
              };
            }
          break;

          case fadeStates::FADING_OUT:
            //nothing to see here
          break;
        }


        

        //adjust position
        mSegmentQ[segIdx].position+= mSegmentQ[segIdx].speed;
        //limit
        if (mSegmentQ[segIdx].position >= mSegmentQ[segIdx].segmentAudio.size()) {
          mSegmentQ[segIdx].position = mSegmentQ[segIdx].segmentAudio.size()-1;
        }else if (mSegmentQ[segIdx].position < 0){
          mSegmentQ[segIdx].position=0;
        }
        //loop if necessary
        // if (mSegmentQ[segIdx].position >= mSegmentQ[segIdx].segmentAudio.size()) {
        //   mSegmentQ[segIdx].position -= mSegmentQ[segIdx].segmentAudio.size();
        // }else if (mSegmentQ[segIdx].position < 0){
        //   mSegmentQ[segIdx].position += mSegmentQ[segIdx].segmentAudio.size();
        // }
        
        audioOut += seqmentSample;
      }

    }else{
      audioOut = 0;
    }
    
    //clear out old segments
    for (auto it = mSegmentQ.begin(); it != mSegmentQ.end(); ) {
        if (it->fadeState == fadeStates::FADING_OUT && it->amp <= 0) {
            // cout << "Clearing seg, size: " << mSegmentQ.size() << ", amp: " << it->amp << endl;
            it = mSegmentQ.erase(it);
        } else {
            ++it;
        }
    }

    
    mSampleClock++;
    return audioOut;
  }

  index getLatency() { return 0; }



private:
  enum fadeStates {FADING_IN, MID, FADING_OUT};
  struct soundSegment {
    RingBuf<double>::winBufType segmentAudio;
    double position;
    double ampMod;
    double amp;
    double fadeLenSamples;
    fadeStates fadeState;
    double speed;

    void startFadeIn() {
        ampMod = 1.0 / fadeLenSamples;
        amp = 0.0;
        fadeState = fadeStates::FADING_IN;
    }

    void startFadeOut() {
      ampMod = -1.0 / fadeLenSamples;
      fadeState = fadeStates::FADING_OUT;
    }
  };
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
  std::deque<soundSegment> mSegmentQ;

  std::random_device rd;
  std::mt19937 mtrand{rd()};

};



} // namespace algorithm
} // namespace fluid



/*
2. speed adjust for fadelen
3. pos adjust for -ve speed, init pos
4. random choice of x closest matches
*/