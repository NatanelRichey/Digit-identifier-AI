#ifndef DENSE_H
#define DENSE_H

#include "Activation.h"

// Insert Dense class here...
class Dense
{
 public:

  /**
  * Inits a new layer with given parameters.
  * Constructor accepts 2 matrices and activation function
  */
  Dense (Matrix weights, Matrix bias, ActivationFunction function);
  Matrix get_weights () const
  { return _weights; }
  Matrix get_bias () const
  { return _bias; }
  ActivationFunction get_activation () const
  { return _function; }
  /**
  * Calculates vector output by weight matrix, bias vector and the activation
  * function.
  */
  Matrix operator() (const Matrix &in) const;

 private:
  Matrix _weights;
  Matrix _bias;
  ActivationFunction _function;
};

#endif //DENSE_H
