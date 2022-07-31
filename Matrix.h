// Matrix.h
#ifndef MATRIX_H
#define MATRIX_H
#include <iostream>

using std::ostream;
using std::istream;

/**
 * @struct matrix_dims
 * @brief Matrix dimensions container. Used in MlpNetwork.h and main.cpp
 */
typedef struct matrix_dims
{
    int rows, cols;
} matrix_dims;

// Insert Matrix class here...
class Matrix
{
 public:
  /**
  * Constructs Matrix of size rows×cols.
  * Inits all elements to 0.
  */
  Matrix (int rows, int cols);
  Matrix () : Matrix (1, 1) {};
  // See full documentation in header file
  /**
  * Copy-Constructor. Constructs matrix from another Matrix m.
  */
  Matrix (Matrix const &m1); // copy constructor
  ~Matrix ();
  int get_rows () const { return this->_rows; }
  int get_cols () const { return this->_cols; }
  /**
  * Transforms a matrix into its transpose matrix, i.e (A.transpose())ij = Aji.
  * Supports function calling concatenation
  */
  Matrix &transpose ();
  /**
  * Transforms a matrix into a column vector.
  * Supports function calling concatenation.
  */
  Matrix &vectorize ();
  /**
  * Prints matrix elements, no return value.
  * Prints space after each element (including last element in row).
  * Prints newline after each row (including last row).
  */
  void plain_print () const;
  /**
  * Returns a matrix which is the elementwise multiplication(Hadamard
  * product) of this matrix and another matrix m.
  */
  Matrix dot (const Matrix &m) const;
  /**
  * Returns the Frobenius norm of the given matrix.
  */
  float norm ();
  Matrix operator+ (const Matrix &b) const;
  /**
  * Default copy - initializer. Deep copies structure.
  */
  Matrix &operator= (const Matrix &b); // copy-assign
  /**
  * Matrix multiplication. Matrix a, b;→ a * b.
  */
  Matrix operator* (const Matrix &b) const;
  /**
  * Scalar multiplication on the right. Matrix m, Scalar c; m * c.
  */
  Matrix operator* (float c) const;
  /**
  * Scalar multiplication on the left. Scalar c, Matrix m; c * m.
  */
  friend Matrix operator* (float c, const Matrix &a);
  Matrix &operator+= (const Matrix &b);
  /**
  * For i,j indices, Matrix m: m(i,j) will return the i,j element in the
  * matrix.
  */
  float operator() (int i, int j) const;
  /**
  * For i,j indices, Matrix m: m(i,j) will assign the i,j element in the
  * matrix.
  */
  float &operator() (int i, int j);
  /**
  * For i index, Matrix m: m[i] will return the i'th element.
  */
  float operator[] (int i) const;
  /**
  * For i index, Matrix m: m[i] will set the i'th element.
  */
  float &operator[] (int i);
  /**
  * Pretty export of matrix.
  */
  friend ostream &operator<< (ostream &os, const Matrix &m);
  /**
  * Fills matrix elements: has to read input stream fully, otherwise
  * it's an error (exception is thrown in the case of error).
  */
  friend istream &operator>> (istream &is, Matrix &m);

 private:
  int _rows, _cols;
  float **_matrix_arr;

  void delete_matrix_array ();
  void shallow_copy_matrix_array (Matrix &tmp_m) const;
  /**
   * Allocates matrix data structure and initializes all elements to 0.
   */
  void alloc_matrix_array ();
  void assert_matrix_dim_err (const Matrix &m) const;
};

#endif //MATRIX_H