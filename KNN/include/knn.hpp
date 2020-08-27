//
// Created by amirt01 on 8/23/20.
//

#ifndef KNN_KNN_HPP
#define KNN_KNN_HPP

#include "../../include/CommonAlg.hpp"

class knn : public CommonAlg {
  int m_k{};
  std::vector<Data*> m_neighbors{};

  double calculate(std::vector<Data*>* set) override;

 public:
  knn() = default;

  explicit knn(const int& val);

  explicit knn(DataHandler* dh, const int& val);

  void findKnearest(Data* queryPoint);

  void setK(const int& val);

  int predict();

  double validatePerformance();

  double testPerformance();
};

#endif //KNN_KNN_HPP
