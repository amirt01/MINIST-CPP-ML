//
// Created by amirt01 on 8/24/20.
//

#include "../include/CommonAlg.hpp"

// initialize data set pointers from DataHandler
CommonAlg::CommonAlg(DataHandler* dh)
  : m_trainingData(dh->getTrainingData()),
    m_testData(dh->getTestData()),
    m_validationData(dh->getValidationData()) {}

void CommonAlg::setTrainingData(std::vector<Data*>* vect) {
  m_trainingData = vect;
}

void CommonAlg::setTestData(std::vector<Data*>* vect) {
  m_testData = vect;
}

void CommonAlg::setValidationData(std::vector<Data*>* vect) {
  m_validationData = vect;
}
