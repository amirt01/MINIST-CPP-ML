//
// Created by amirt01 on 8/24/20.
//

#include "KMeans.hpp"

// for io
#include <iostream>
#include <iomanip>

// for random splitting of m_validationData and m_testData
#include <random>
#include <chrono>

// for initializing a cluster for each class
#include <unordered_set>

double KMeans::calculate(std::vector<Data*>* set) {
  double numCorrect = 0.0;  // initialize value to keep track of the number of correct predictions
  for (auto queryPoint : *set) {  // for each data point in the data set
    // initialize the closest cluster to be the maximum possible distance
    double minDist = std::numeric_limits<double>::max();
    cluster* bestCluster = nullptr;  // initialize the best cluster to be a random cluster
    for (auto& cluster : m_clusters) {  // for each cluster in m_clusters
      double currentDist = queryPoint->calculateDistance(cluster.m_centroid);
      if (currentDist < minDist) {
        minDist = currentDist;  // update the minimum distance
        bestCluster = &cluster;  // update hte best cluster
      }
    }
    // if the bestCluster's m_mostFrequentClass is the same as the queryPoint's label
    if (bestCluster->m_mostFrequentClass == queryPoint->getLabel()) {
      ++numCorrect;  // add one to numCorrect
    }
  }
  return 100.0 * (numCorrect / static_cast<double>(set->size()));  // return the percent correct
}

KMeans::KMeans(const int& k)
  : m_numClusters(k) {}

KMeans::KMeans(DataHandler* dh, const int& k)
  : CommonAlg(dh), m_numClusters(k) {}

void KMeans::initClusters() {
  // initialize a vector to hold the random data
  std::vector<Data*> randomData;

  // take a random m_numClusters samples from m_trainingData
  std::sample(m_trainingData->begin(), m_trainingData->end(), std::back_inserter(randomData), m_numClusters,
              std::mt19937{std::random_device{}()});

  // add the random sample to m_clusters
  for (auto* data : randomData) {
    m_clusters.emplace_back(data);
  }
}

void KMeans::initClustersForEachClass() {
  std::unordered_set<int> classesUsed;
  for (auto& data : *m_trainingData) {  // for each data in m_trainingData
    // test to see if it's class is already in classesUsed
    if (classesUsed.find(data->getLabel()) == classesUsed.end()) {
      m_clusters.emplace_back(data);  // use it as a cluster node
      classesUsed.insert(data->getLabel());  // add it's class to classesUsed
    }
  }
}

void KMeans::train() {
  for (auto* data : *m_trainingData) {  // for each data point in the training data set
    // initialize the closest cluster to be the maximum possible distance
    double minDist = std::numeric_limits<double>::max();
    cluster* bestCluster = nullptr;  // initialize a pointer to keep track of the closest cluster
    for (auto& cluster : m_clusters) {  // for each cluster in m_clusters
      // calculate the distance from data to the cluster's m_centroid
      double currentDist = data->calculateDistance(cluster.m_centroid);
      // if the data point is closer to this cluster than the others that have been checked
      if (currentDist < minDist) {
        minDist = currentDist;  // update the minimum distance
        bestCluster = &cluster;  // update hte best cluster
      }
    }
    bestCluster->addToCluster(data);  // add data to the best cluster found
  }
}

double KMeans::validate() {
  return calculate(m_validationData);
}

double KMeans::test() {
  return calculate(m_testData);
}


int main() {
  // initialize the data handler and fill all of it's classes
  DataHandler dh;
  dh.readFeatureVector("../data/train-images-idx3-ubyte");
  dh.readFeatureLabels("../data/train-labels-idx1-ubyte");
  dh.splitData();
  dh.countClasses();

  int k = static_cast<int>(dh.getTestData()->size() * 0.1);
  KMeans km(&dh, k);
  km.initClusters();
  km.train();
  std::cout << "Tested Performance for K = " << k << ": " << std::fixed << std::setprecision(2) << km.test()
            << "%\n";
  std::cout << "Validated Performance for K = " << k << ": " << std::fixed << std::setprecision(2) << km.validate()
            << "%\n";
}
