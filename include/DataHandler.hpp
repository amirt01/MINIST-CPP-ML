//
// Created by amirt01 on 8/21/20.
//

#ifndef MNIST_HANDWRITING_DATA_ML_DATAHANDLER_HPP
#define MNIST_HANDWRITING_DATA_ML_DATAHANDLER_HPP

#include <string>
#include <map>

#include "Data.hpp"

// constants from the MINIST Database website: http://yann.lecun.com/exdb/mnist/
constexpr uint32_t LABEL_MAGIC = 0x00000801;  // 2049
constexpr uint32_t IMAGE_MAGIC = 0x00000803;  // 2051

constexpr uint32_t NUM_TRAINING = 60000;
constexpr uint32_t NUM_TEST = 10000;

// calculated sizes for each data set
constexpr unsigned TRAIN_SET_SIZE = static_cast<unsigned>(NUM_TRAINING * 0.75);  // 75%
constexpr unsigned TEST_SET_SIZE = static_cast<unsigned>(NUM_TRAINING * 0.20);  // 20%
constexpr unsigned VALIDATION_SIZE = static_cast<unsigned>(NUM_TRAINING * 0.05);  // 5%

class DataHandler {
  // arrays to hold all of the Data to be stored
  std::vector<Data> m_dataArray;  // all of the Data (pre-split)
  std::vector<Data*> m_trainingData;
  std::vector<Data*> m_testData;
  std::vector<Data*> m_validationData;

  int m_numClasses{};
  std::map<uint8_t, int> m_classMap;

 public:
  DataHandler();

  void readFeatureVector(const std::string& path);

  void readFeatureLabels(const std::string& path);

  void splitData();

  void countClasses();

  static uint32_t convertToLittleEndian(const uint8_t*);

  [[nodiscard]] int getClassCount() const;

  std::vector<Data*>* getTrainingData();

  std::vector<Data*>* getTestData();

  std::vector<Data*>* getValidationData();
};


#endif //MNIST_HANDWRITING_DATA_ML_DATAHANDLER_HPP
