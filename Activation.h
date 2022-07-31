#include "Matrix.h"

#ifndef ACTIVATION_H
#define ACTIVATION_H

typedef Matrix (*ActivationFunction) (const Matrix &in);

namespace activation
{
    /**
    * Activation function. Function returns 0 if x < 0; else x. Function is
    * applied to each coordinate of vector.
    * Used in first three layers.
    */
    Matrix softmax (const Matrix &in);

    /**
    * Activation function. Return distributed vector where all elements sum to 1.
    * Used in last layer.
    */
    Matrix relu (const Matrix &in);
}

#endif //ACTIVATION_H