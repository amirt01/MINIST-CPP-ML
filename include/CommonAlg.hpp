//
// Created by amirt01 on 8/24/20.
//

#ifndef MNIST_ML_COMMONALG_HPP
#define MNIST_ML_COMMONALG_HPP

#include "DataHandler.hpp"

class CommonAlg {
 protected:
  std::vector<Data*>* m_trainingData{};
  std::vector<Data*>* m_testData{};
  std::vector<Data*>* m_validationData{};

  virtual double calculate(std::vector<Data*>*) = 0;

 public:
  CommonAlg() = default;

  explicit CommonAlg(DataHandler* dh);

  void setTrainingData(std::vector<Data*>* vect);

  void setTestData(std::vector<Data*>* vect);

  void setValidationData(std::vector<Data*>* vect);
};


#endif //MNIST_ML_COMMONALG_HPP
