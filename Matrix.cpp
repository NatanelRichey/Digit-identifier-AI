#include <iostream>
#include "Matrix.h"
#include <cmath>

#define MIN_BRIGHTNESS 0.1
#define MATRIX_INPUT_ERROR_MSG "Input Error: please enter positive matrix " \
                            "dimensions."
#define MATRIX_DIM_ERROR_MSG_1 "Dimension Error: please provide matrices of " \
                            "similar dimensions."
#define MATRIX_DIM_ERROR_MSG_2 "Dimension Error: please provide matrices " \
                            "defined for matrix multiplication."
#define ACCESS_ERROR_MSG "Access Error: please provide a valid coordinate."
#define FILE_ERROR_MSG_1 "File Error: please provide a valid file."
#define FILE_ERROR_MSG_2 "File Error: file could not be read."

using std::runtime_error;
using std::length_error;
using std::out_of_range;


using std::ostream;
using std::istream;
using std::cout;
using std::ifstream;

// See full documentation in header file
Matrix::Matrix (int rows, int cols)
{
  _rows = rows;
  _cols = cols;
  if (_rows <= 0 || _cols <= 0)
  { throw length_error (MATRIX_INPUT_ERROR_MSG); }
  alloc_matrix_array ();
}

// See full documentation in header file
void Matrix::alloc_matrix_array ()
{
  _matrix_arr = new float *[_rows];
  for (int i = 0; i < _rows; i++)
  {
    _matrix_arr[i] = new float[_cols];
    for (int j = 0; j < _cols; j++)
    {
      _matrix_arr[i][j] = 0;
    }
  }
}

// See full documentation in header file
Matrix::Matrix (Matrix const &m1)
{
  _rows = 0;
  _cols = 0;
  _matrix_arr = nullptr;
  this->operator= (m1);
}

Matrix::~Matrix ()
{
  delete_matrix_array ();
}

void Matrix::delete_matrix_array ()
{
  for (int i = 0; i < _rows; i++)
  {
    delete[] _matrix_arr[i];
    _matrix_arr[i] = nullptr;
  }
  delete[] _matrix_arr;
  _matrix_arr = nullptr;
}

// See full documentation in header file
Matrix &Matrix::transpose ()
{
  Matrix tmp_m (_rows, _cols);
  shallow_copy_matrix_array (tmp_m);
  delete_matrix_array ();
  _rows = tmp_m._cols;
  _cols = tmp_m._rows;
  _matrix_arr = new float *[_rows];
  for (int i = 0; i < _rows; i++)
  {
    _matrix_arr[i] = new float[_cols];
    for (int j = 0; j < _cols; j++)
    {
      _matrix_arr[i][j] = tmp_m._matrix_arr[j][i];
    }
  }
  return *this;
}

void Matrix::shallow_copy_matrix_array (Matrix &tmp_m) const
{
  for (int i = 0; i < _rows; i++)
  {
    for (int j = 0; j < _cols; j++)
    {
      tmp_m._matrix_arr[i][j] = _matrix_arr[i][j];
    }
  }
}

// See full documentation in header file
Matrix &Matrix::vectorize ()
{
  Matrix tmp_m (_rows, _cols);
  shallow_copy_matrix_array (tmp_m);
  delete_matrix_array ();
  int counter = 0;
  _rows = tmp_m._cols * tmp_m._rows;
  _cols = 1;
  alloc_matrix_array ();
  for (int i = 0; i < tmp_m._rows; i++)
  {
    for (int j = 0; j < tmp_m._cols; j++)
    {
      _matrix_arr[counter][0] = tmp_m._matrix_arr[i][j];
      counter++;
    }
  }
  return *this;
}

// See full documentation in header file
void Matrix::plain_print () const
{
  for (int i = 0; i < _rows; i++)
  {
    for (int j = 0; j < _cols; j++)
    {
      cout << _matrix_arr[i][j] << ' ';
    }
    cout << std::endl;
  }
}

// See full documentation in header file
Matrix Matrix::dot (const Matrix &m) const
{
  assert_matrix_dim_err (m);
  Matrix dot_m (_rows, _cols);
  for (int i = 0; i < _rows; i++)
  {
    for (int j = 0; j < _cols; j++)
    {
      dot_m._matrix_arr[i][j] = this->_matrix_arr[i][j] * m._matrix_arr[i][j];
    }
  }
  return dot_m;
}

void Matrix::assert_matrix_dim_err (const Matrix &m) const
{
  if (_rows != m._rows || _cols != m._cols)
  {
    throw length_error (MATRIX_DIM_ERROR_MSG_1);
  }
}

// See full documentation in header file
float Matrix::norm ()
{
  float sigma = 0, elem_sq;
  for (int i = 0; i < _rows; i++)
  {
    for (int j = 0; j < _cols; j++)
    {
      elem_sq = _matrix_arr[i][j] * _matrix_arr[i][j];
      sigma += elem_sq;
    }
  }
  return sqrt (sigma);
}

Matrix Matrix::operator+ (const Matrix &b) const
{
  assert_matrix_dim_err (b);
  Matrix sum_m (_rows, _cols);
  for (int i = 0; i < _rows; i++)
  {
    for (int j = 0; j < _cols; j++)
    {
      sum_m._matrix_arr[i][j] = _matrix_arr[i][j] + b._matrix_arr[i][j];
    }
  }
  return sum_m;
}

// See full documentation in header file
Matrix &Matrix::operator= (const Matrix &b)
{
  if (&b == this)
  { return *this; }
  delete_matrix_array ();
  _rows = b._rows;
  _cols = b._cols;
  _matrix_arr = new float *[_rows];
  for (int i = 0; i < _rows; i++)
  {
    _matrix_arr[i] = new float[_cols];
    for (int j = 0; j < _cols; j++)
    {
      _matrix_arr[i][j] = b._matrix_arr[i][j];
    }
  }
  return *this;
}

// See full documentation in header file
Matrix Matrix::operator* (const Matrix &b) const
{
  if (_cols != b._rows)
  { throw length_error (MATRIX_DIM_ERROR_MSG_2); }
  Matrix mult_m (_rows, b._cols);
  for (int i = 0; i < _rows; ++i)
  {
    for (int j = 0; j < b._cols; ++j)
    {
      for (int k = 0; k < _cols; ++k)
      {
        mult_m._matrix_arr[i][j] += _matrix_arr[i][k] * b._matrix_arr[k][j];
      }
    }
  }
  return mult_m;
}

// See full documentation in header file
Matrix Matrix::operator* (const float c) const
{
  Matrix mult_m (_rows, _cols);
  for (int i = 0; i < _rows; i++)
  {
    for (int j = 0; j < _cols; j++)
    {
      mult_m._matrix_arr[i][j] = _matrix_arr[i][j] * c;
    }
  }
  return mult_m;
}

// See full documentation in header file
Matrix operator* (const float c, const Matrix &a)
{
  Matrix mult_m (a._rows, a._cols);
  for (int i = 0; i < a._rows; i++)
  {
    for (int j = 0; j < a._cols; j++)
    {
      mult_m._matrix_arr[i][j] = a._matrix_arr[i][j] * c;
    }
  }
  return mult_m;
}

Matrix &Matrix::operator+= (const Matrix &b)
{
  assert_matrix_dim_err (b);
  for (int i = 0; i < _rows; i++)
  {
    for (int j = 0; j < _cols; j++)
    {
      _matrix_arr[i][j] += b._matrix_arr[i][j];
    }
  }
  return *this;
}

// See full documentation in header file
float Matrix::operator() (const int i, const int j) const
{
  if (i >= _rows || i < 0 || j >= _cols || j < 0)
  {
    throw out_of_range
        (ACCESS_ERROR_MSG);
  }
  return _matrix_arr[i][j];
}

// See full documentation in header file
float &Matrix::operator() (const int i, const int j)
{
  if (i >= _rows || i < 0 || j >= _cols || j < 0)
  {
    throw out_of_range
        (ACCESS_ERROR_MSG);
  }
  return _matrix_arr[i][j];
}

// See full documentation in header file
float Matrix::operator[] (const int i) const
{
  if (i >= _rows * _cols || i < 0)
  { throw out_of_range (ACCESS_ERROR_MSG); }
  int row = i / _cols;
  int col = i % _cols;
  return _matrix_arr[row][col];
}

// See full documentation in header file
float &Matrix::operator[] (const int i)
{
  if (i >= _rows * _cols || i < 0)
  { throw out_of_range (ACCESS_ERROR_MSG); }
  int row = i / _cols;
  int col = i % _cols;
  return _matrix_arr[row][col];
}

// See full documentation in header file
ostream &operator<< (ostream &os, const Matrix &m)
{
  for (int i = 0; i < m._rows; i++)
  {
    for (int j = 0; j < m._cols; j++)
    {
      if (m._matrix_arr[i][j] > MIN_BRIGHTNESS)
      { cout << "**"; }
      else
      { cout << "  "; }
    }
    cout << std::endl;
  }
  return os;
}

// See full documentation in header file
istream &operator>> (istream &is, Matrix &a)
{
  float length;
  is.seekg (0, std::ios::end);
  length = is.tellg ();
  if (length != a._cols * a._rows * sizeof (float))
  {
    throw runtime_error
        (FILE_ERROR_MSG_1);
  }
  is.seekg (0, std::ios::beg);
  for (int i = 0; i < a._rows; ++i)
  {
    if (!is.read ((char *) a._matrix_arr[i], length / a._rows))
    {
      throw runtime_error (FILE_ERROR_MSG_2);
    }
  }
  return is;
}

