//
// Created by amirt01 on 8/24/20.
//

#ifndef MNIST_ML_KMEANS_HPP
#define MNIST_ML_KMEANS_HPP

#include "../../include/CommonAlg.hpp"
#include "../../include/DataHandler.hpp"

typedef struct cluster {
  std::vector<uint8_t> m_centroid;
  std::vector<Data*> m_clusterPoints{};
  std::map<int, int> m_classCounts{};
  int m_mostFrequentClass{};

  // initialize the cluster based on an initialPoint
  explicit cluster(Data* initialPoint)
    : m_centroid(*initialPoint->getFeatureVector()) {
    m_clusterPoints.push_back(initialPoint);
    m_classCounts[initialPoint->getLabel()] = 1;
    m_mostFrequentClass = initialPoint->getLabel();
  }

  // add a point to a cluster
  void addToCluster(Data* point) {
    size_t previousSize = m_clusterPoints.size();  // store the previous cluster size
    m_clusterPoints.push_back(point);  // add the point to this cluster
    for (int i = 0; i < m_centroid.size() - 1; ++i) {  //
      double value = m_centroid.at(i);
      value *= previousSize;
      value += point->getFeatureVector()->at(i);
      value /= static_cast<double>(m_clusterPoints.size());
      m_centroid.at(i) = value;
    }
    if (m_classCounts.find(point->getLabel()) == m_classCounts.end()) {
      m_classCounts[point->getLabel()] = 1;
    } else {
      m_classCounts[point->getLabel()]++;
    }
  }

  void setMostFrequentClass() {
    int bestClass = 0;
    int freq = 0;
    for (auto kv : m_classCounts) {
      if (kv.second > freq) {
        freq = kv.second;
        bestClass = kv.first;
      }
    }
    m_mostFrequentClass = bestClass;
  }
} cluster_t;

class KMeans : public CommonAlg {
  int m_numClusters;
  std::vector<cluster_t> m_clusters{};

  double calculate(std::vector<Data*>* set) override;

 public:
  explicit KMeans(const int& k);

  KMeans(DataHandler* dh, const int& k);

  void initClusters();

  void initClustersForEachClass();

  void train();

  double validate();

  double test();
};


#endif //MNIST_ML_KMEANS_HPP
