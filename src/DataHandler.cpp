//
// Created by amirt01 on 8/21/20.
//

// for io
#include <iostream>
#include <fstream>
#include <iomanip>

// for random splitting of m_dataArray
#include <unordered_set>
#include <random>
#include <chrono>

#include "../include/DataHandler.hpp"

// fill m_dataArray with empty data Data objects to be used by readFeatureVector and readFeatureLabels
// allocate space for each of the other vectors to be split into later
DataHandler::DataHandler()
  : m_dataArray(NUM_TRAINING, Data{}), m_trainingData(TRAIN_SET_SIZE),
    m_testData(TEST_SET_SIZE), m_validationData(VALIDATION_SIZE) {}

void DataHandler::readFeatureVector(const std::string& path) {
  std::array<uint32_t, 4> header{};  // |MAGIC|NUM IMAGES|ROWSIZE|COLSIZE|
  uint8_t bytes[4];
  // open the MNIST Database file
  std::ifstream fin(path, std::ios::in | std::ios::binary);
  // test if the file is open correctly
  if (fin.is_open()) {
    // initialize the header from the file
    for (auto& val : header) {
      // read in each value from the binary file
      if (fin.read(reinterpret_cast<char*>(bytes), sizeof(bytes))) {
        // store the value in the header array
        val = convertToLittleEndian(bytes);
      } else {
        std::cout << "Error Reading from File.\n";
        exit(EXIT_FAILURE);
      }
    }

    if (header[0] != IMAGE_MAGIC) {  // check magic number
      std::cout << "Error Reading from File: invalid magic number.\n";
      exit(EXIT_FAILURE);
    } else if (header[1] != NUM_TRAINING) {  // check the number of images in this file
      std::cout << "Error Reading from File: invalid number of images.\n";
      exit(EXIT_FAILURE);
    } else if (header[2] != header[3] || IMAGE_SIZE != (header[2] * header[3])) {  // check the image size
      std::cout << "Error Reading from File: invalid image size.\n";
      exit(EXIT_FAILURE);
    } else {
      std::cout << "File validation complete.\n";
    }

    // for each image in the file (based on value from header)
    for (auto& i : m_dataArray) {
      // for each byte in an image
      for (auto& j : *i.getFeatureVector()) {
        // read in the byte from the image
        if (!fin.read(reinterpret_cast<char*>(&j), sizeof(j))) {
          std::cout << "Error Reading from File.\n";
          exit(EXIT_FAILURE);
        }
      }
    }
    std::cout << "Successfully read and stored " << m_dataArray.size() << " feature vectors\n";
  } else {
    // if the file wasn't opened correctly, exit the program
    std::cout << "Could not find file.\n";
    exit(EXIT_FAILURE);
  }
}

void DataHandler::readFeatureLabels(const std::string& path) {
  std::array<uint32_t, 2> header{};  // |MAGIC|NUM IMAGES|
  uint8_t bytes[4];
  // open the MNIST Database file
  std::ifstream fin(path, std::ios::in | std::ios::binary);
  // test if the file is open correctly
  if (fin.is_open()) {
    // initialize the header from the file
    for (auto& val : header) {
      // read in each value from the binary file
      if (fin.read(reinterpret_cast<char*>(bytes), sizeof(bytes))) {
        // store the value in the header array
        val = convertToLittleEndian(bytes);
      }
    }
    std::cout << "Done getting label file header.\n";

    if (header[0] != LABEL_MAGIC) {  // check magic number
      std::cout << "Error Reading from File: invalid magic number.\n";
      exit(EXIT_FAILURE);
    } else if (header[1] != NUM_TRAINING) {  // check the number of images in this file
      std::cout << "Error Reading from File: invalid number of images.\n";
      exit(EXIT_FAILURE);
    } else {
      std::cout << "File validation complete.\n";
    }

    // for each image in the file (based on value from header)
    for (int i = 0; i < header[1]; ++i) {
      uint8_t element[1];
      if (fin.read(reinterpret_cast<char*>(element), sizeof(element))) {
        m_dataArray.at(i).setLabel(*element);
      } else {
        std::cout << "Error Reading from File.\n";
        exit(EXIT_FAILURE);
      }
    }
    std::cout << "Successfully read and stored " << m_dataArray.size() << " labels\n";
  } else {
    std::cout << "Could not find file.\n";
    exit(EXIT_FAILURE);
  }
}

void DataHandler::splitData() {
  // unordered_set to keep track of the indexes already across all of the data sets
  std::unordered_set<int> usedIndexes;

  // set up random number generation
  long seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine rand(seed);

  // lambda function to uniquely split m_dataArray into a vector of Data*
  auto splitter = [&](std::vector<Data*>* data) {
    int count = 0;
    while (count < data->size()) {
      int randIndex = static_cast<int>(rand() % m_dataArray.size()); // 0 & m_dataArray->size() - 1
      // only add the item if the index has not been used already
      if (usedIndexes.find(randIndex) == usedIndexes.end()) {
        data->at(count) = &m_dataArray.at(randIndex);
        usedIndexes.insert(randIndex);
        ++count;
      }
    }
  };

  // uniquely split m_dataArray into each data set
  splitter(&m_trainingData);
  splitter(&m_testData);
  splitter(&m_validationData);

  std::cout << "Training Data Size: " << m_trainingData.size() << '\n';
  std::cout << "Test Data Size: " << m_testData.size() << '\n';
  std::cout << "Validation Data Size: " << m_validationData.size() << '\n';
}

void DataHandler::countClasses() {
  m_numClasses = 0;  // reset the current m_numClasses value
  // iterate through all the data in m_dataArray
  for (auto& data : m_dataArray) {
    // count a class if the current data is not found in m_classMap
    if (m_classMap.find(data.getLabel()) == m_classMap.end()) {
      m_classMap[data.getLabel()] = m_numClasses;  // add the current data to m_classMap
      data.setEnumeratedLabel(m_numClasses);  // store the enumeratedLabel in the data
      ++m_numClasses; // increase the count
    }
  }
  std::cout << "Successfully extracted " << m_numClasses << " unique classes.\n";
}

uint32_t DataHandler::convertToLittleEndian(const uint8_t* bytes) {
  return bytes[0] << 24U |
         bytes[1] << 16U |
         bytes[2] << 8U |
         bytes[3];
}

int DataHandler::getClassCount() const {
  return m_numClasses;
}

std::vector<Data*>* DataHandler::getTrainingData() {
  return &m_trainingData;
}

std::vector<Data*>* DataHandler::getTestData() {
  return &m_testData;
}

std::vector<Data*>* DataHandler::getValidationData() {
  return &m_validationData;
}

/*
int main() {
  DataHandler dh;
  dh.readFeatureVector("../data/train-images-idx3-ubyte");
  dh.readFeatureLabels("../data/train-labels-idx1-ubyte");
  dh.splitData();
  dh.countClasses();

  return EXIT_SUCCESS;
}
*/