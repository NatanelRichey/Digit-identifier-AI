#include "MlpNetwork.h"

// See full documentation in header file
MlpNetwork::MlpNetwork (Matrix *weights, Matrix *biases) :
    _layer1 (weights[0], biases[0], activation::relu),
    _layer2 (weights[1], biases[1], activation::relu),
    _layer3 (weights[2], biases[2], activation::relu),
    _layer4 (weights[3], biases[3], activation::softmax)
{
  _weights[0] = weights[0];
  _weights[1] = weights[1];
  _weights[2] = weights[2];
  _weights[3] = weights[3];
  _biases[0] = biases[0];
  _biases[1] = biases[1];
  _biases[2] = biases[2];
  _biases[3] = biases[3];
}

// See full documentation in header file
digit MlpNetwork::operator() (const Matrix &img) const
{
  const Matrix r1 = _layer1 (img);
  const Matrix r2 = _layer2 (r1);
  const Matrix r3 = _layer3 (r2);
  const Matrix r4 = _layer4 (r3);
  digit digit;
  digit.probability = 0;
  for (int i = 0; i < r4.get_rows (); i++)
  {
    if (r4 (i, 0) > digit.probability)
    {
      digit.probability = r4 (i, 0);
      digit.value = i;
    }
  }
  return digit;
}