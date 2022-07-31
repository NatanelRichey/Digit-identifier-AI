//MlpNetwork.h

#ifndef MLPNETWORK_H
#define MLPNETWORK_H

#include "Dense.h"

#define MLP_SIZE 4

/**
 * @struct digit
 * @brief Identified (by Mlp network) digit with
 *        the associated probability.
 * @var value - Identified digit value
 * @var probability - identification probability
 */
typedef struct digit
{
    unsigned int value;
    float probability;
} digit;

const matrix_dims img_dims = {28, 28};
const matrix_dims weights_dims[] = {{128, 784},
                                    {64,  128},
                                    {20,  64},
                                    {10,  20}};
const matrix_dims bias_dims[] = {{128, 1},
                                 {64,  1},
                                 {20,  1},
                                 {10,  1}};

// Insert MlpNetwork class here...
class MlpNetwork
{
 public:
  /**
  * Accepts 2 arrays of matrices, size 4 each. one for weights and one for
  * biases.
  */
  MlpNetwork (Matrix weights[], Matrix biases[]);
  /**
  * Constructs the network described and applies the entire network on input.
  * @return digit struct.
  */
  digit operator() (const Matrix &img) const;

 private:
  Matrix _weights[MLP_SIZE];
  Matrix _biases[MLP_SIZE];
  Dense _layer1, _layer2, _layer3, _layer4;
};

#endif // MLPNETWORK_H