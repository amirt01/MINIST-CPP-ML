//
// Created by amirt01 on 8/21/20.
//

#ifndef MNIST_HANDWRITING_DATA_ML_DATA_HPP
#define MNIST_HANDWRITING_DATA_ML_DATA_HPP

#include <vector>
#include <cstdint>

// constants from the MINIST Database website: http://yann.lecun.com/exdb/mnist/
constexpr static unsigned IMAGE_SIZE = 784;  // 28x28

class Data {
  std::vector<uint8_t> m_featureVector{};  // No class at end.
  uint8_t m_label{};
  int m_enumLabel{};
  double m_distance{};

 public:
  Data();

  void setLabel(uint8_t byte);

  void setEnumeratedLabel(int label);

  void setDistance(double val);

  [[nodiscard]] double calculateDistance(const Data& input) const;

  [[nodiscard]] double calculateDistance(const std::vector<uint8_t>&) const;

  int getFeatureVectorSize();

  [[nodiscard]] uint8_t getLabel() const;

  [[nodiscard]] int getEnumeratedLabel() const;

  [[nodiscard]] double getDistance() const;

  std::vector<uint8_t>* getFeatureVector();
};


#endif //MNIST_HANDWRITING_DATA_ML_DATA_HPP
