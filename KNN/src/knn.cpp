//
// Created by amirt01 on 8/23/20.
//

#include "knn.hpp"

#include <iostream>
#include <iomanip>
#include <limits>
#include <map>
#include <cstdint>
#include "../../include/DataHandler.hpp"

double knn::calculate(std::vector<Data*>* set) {
  int count = 0;  // a count of the number of correct predictions
  int index = 0;  // the current index in the data set
  for (auto& queryPoint : *set) {  // for each queryPoint in the data set
    findKnearest(queryPoint);  // find the k nearest points to the queryPoint
    int prediction = predict();
    std::cout << prediction << " -> " << unsigned(queryPoint->getLabel()) << '\n';
    if (prediction == queryPoint->getLabel()) {
      ++count;
    }
    // current performance = count * 100 / index
    std::cout << "Test Performance: " << std::fixed << std::setprecision(3)
              << (static_cast<double>(count) * 100.0) / static_cast<double>(++index) << "%\n";
  }
  // validation performance = count * 100 / size of the data set
  double testPerformance = (static_cast<double>(count) * 100.0) / static_cast<double>(set->size());
  std::cout << "Test Performance = " << std::fixed << std::setprecision(3) << testPerformance << "%\n";
  return testPerformance;
}

knn::knn(const int& val) : m_k(val) {}

// initialize all of our data sets using their respective DataHandler function
knn::knn(DataHandler* dh, const int& val)
  : CommonAlg(dh), m_k(val) {}


void knn::findKnearest(Data* queryPoint) {
  // set the initial min_dist / previous_min_dist to be the maximum possible distance
  double min_dist = std::numeric_limits<double>::max();
  double previous_min_dist = min_dist;
  // create a pointer to keep track of the current closest value we want to append to m_neighbors
  Data* closest_ptr = nullptr;
  // find the m_k closest features
  for (int i = 0; i < m_k; ++i) {
    if (i == 0) {  // for the first value
      for (auto& feature : *m_trainingData) {  // check each feature in m_trainingData
        // calculate the distance from the queryPoint to that value
        double distance = queryPoint->calculateDistance(*feature);
        feature->setDistance(distance);  // store the calculated distance for future use
        if (distance < min_dist) {  // check if this is the minimum distance
          min_dist = distance;  // store the minimum distance
          closest_ptr = feature;  // store the closest feature
        }
      }
    } else {  // for every value after the first value
      for (auto& feature : *m_trainingData) {  // check each feature in m_trainingData
        // collect the previously stored distance so we don't need to recalculate it
        double distance = feature->getDistance();
        // only update closest_ptr if this is farther than the previous closest
        if (distance > previous_min_dist && distance < min_dist) {
          min_dist = distance;  // store the minimum distance
          closest_ptr = feature;  // store the closest feature
        }
      }
    }
    m_neighbors.push_back(closest_ptr);  // append the closest feature
    previous_min_dist = min_dist;  // keep track of the previous min
    min_dist = std::numeric_limits<double>::max();  // reset the min value
  }
}

void knn::setK(const int& val) {
  m_k = val;
}

int knn::predict() {
  std::map<uint8_t, int> classFreq;  // hold a list of each class and the number of classes associated with it
  for (auto& feature : m_neighbors) {  // cycle through each feature in m_neighbors
    if (classFreq.find(feature->getLabel()) == classFreq.end()) {  // check if the label is not already in this list
      classFreq[feature->getLabel()] = 1;  // add the class to the map
    } else {
      classFreq[feature->getLabel()]++;  // increase the number of classes by one
    }
  }

  int best = 0;
  int max = 0;

  // iterate through the classFreq map
  for (auto kv : classFreq) {
    // find the most common class
    if (kv.second > max) {
      max = kv.second;
      best = kv.first;
    }
  }

  m_neighbors.clear();  // cleanup m_neighbors now that we're done with it
  return best;
}

double knn::validatePerformance() {
  return calculate(m_validationData);
}

double knn::testPerformance() {
  return calculate(m_testData);
}

int main() {
  DataHandler dh;
  dh.readFeatureVector("../data/train-images-idx3-ubyte");
  dh.readFeatureLabels("../data/train-labels-idx1-ubyte");
  dh.splitData();
  dh.countClasses();

  knn knearest(&dh, 1);
  knearest.validatePerformance();
  knearest.testPerformance();
  return EXIT_SUCCESS;
}
