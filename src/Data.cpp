//
// Created by amirt01 on 8/21/20.
//

#include "../include/Data.hpp"
#include <iostream>
#include <cmath>

Data::Data() : m_featureVector(IMAGE_SIZE) {}

void Data::setLabel(uint8_t byte) {
  m_label = byte;
}

void Data::setEnumeratedLabel(int label) {
  m_enumLabel = label;
}

void Data::setDistance(double val) {
  m_distance = val;
}

int Data::getFeatureVectorSize() {
  return m_featureVector.size();
}

uint8_t Data::getLabel() const {
  return m_label;
}

int Data::getEnumeratedLabel() const {
  return m_enumLabel;
}

double Data::getDistance() const {
  return m_distance;
}

std::vector<uint8_t>* Data::getFeatureVector() {
  return &m_featureVector;
}

double Data::calculateDistance(const Data& input) const {
  return calculateDistance(input.m_featureVector);
}

double Data::calculateDistance(const std::vector<uint8_t>& input) const {
  double dist = 0.0;
  // safety check for
  if (m_featureVector.size() != input.size()) {
    std::cout << "Error: vector size mismatch.\n";
    exit(EXIT_FAILURE);
  }
  // euclidean distance formula: sqrt(a^2 + b^2) = dist
  for (int i = 0; i < m_featureVector.size(); ++i) {
    dist += pow(input.at(i) - m_featureVector.at(i), 2);
  }

  return sqrt(dist);
}