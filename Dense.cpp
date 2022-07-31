#include "Dense.h"

// See full documentation in header file
Dense::Dense (Matrix weights, Matrix bias, ActivationFunction function)
{
  _weights = weights;
  _bias = bias;
  _function = function;
}


// See full documentation in header file
Matrix Dense::operator() (const Matrix &in) const
{
  Matrix out = _function ((_weights * in) + _bias);
  return out;
}