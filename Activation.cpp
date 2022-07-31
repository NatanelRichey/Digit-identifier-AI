#include "Activation.h"
#include <cmath>

// See full documentation in header file
Matrix activation::relu (const Matrix &in)
{
  Matrix out (in.get_rows (), in.get_cols ());
  for (int i = 0; i < in.get_rows (); ++i)
  {
    for (int j = 0; j < in.get_cols (); ++j)
    {
      if (in (i, j) >= 0)
      { out (i, j) = in (i, j); }
      else
      { out (i, j) = 0; }
    }
  }
  return out;
}

// See full documentation in header file
Matrix activation::softmax (const Matrix &in)
{
  float sigma = 0;
  for (int i = 0; i < in.get_rows (); i++)
  {
    float val = in (i, in.get_cols () - 1);
    sigma += exp (val);
  }
  float lambda = 1 / sigma;
  Matrix exp_vector (in.get_rows (), in.get_cols ());
  for (int i = 0; i < in.get_rows (); i++)
  {
    exp_vector (i, in.get_cols () - 1) = exp (in (i, in.get_cols () - 1));
  }
  return lambda * exp_vector;
}

