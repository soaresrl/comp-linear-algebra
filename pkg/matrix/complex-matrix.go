package matrix

import (
	"errors"
	"math"
	"math/cmplx"
)

type ComplexMatrix struct {
	Coef [][]complex128
	Rows int
	Cols int
}

func (mat *ComplexMatrix) FromArray(array [][]complex128) *ComplexMatrix {
	result := &ComplexMatrix{Coef: array, Rows: len(array), Cols: len(array[0])}

	return result
}

func (mat *ComplexMatrix) Copy() *ComplexMatrix {
	dst := ComplexMatrix{Cols: mat.Cols, Rows: mat.Rows}

	for i := 0; i < mat.Rows; i++ {
		for j := 0; j < mat.Cols; j++ {
			dst.Coef[i][j] = mat.Coef[i][j]
		}
	}

	return &dst
}

func (mat *ComplexMatrix) Norm() float64 {
	sum := complex(0, 0)

	for i := 0; i < mat.Rows; i++ {
		sum += mat.Conj().Row(i).Dot(mat.Row(i))
	}

	return math.Sqrt(real(sum))
}

func (mat *ComplexMatrix) Conj() *ComplexMatrix {
	result := mat.Copy()

	for i := 0; i < result.Rows; i++ {
		for j := 0; j < result.Cols; j++ {
			result.Coef[i][j] = cmplx.Conj(result.Coef[i][j])
		}
	}

	return result
}

// Returns the column i as a *Matrix
func (mat *ComplexMatrix) Col(i int) *ComplexMatrix {
	col := &ComplexMatrix{Rows: mat.Rows, Cols: 1}

	for j := 0; j < mat.Rows; j++ {
		col.Coef[j][0] = mat.Coef[j][i]
	}

	return col
}

// Returns the row i as a *Matrix
func (mat *ComplexMatrix) Row(i int) *ComplexMatrix {
	row := &ComplexMatrix{Rows: 1, Cols: mat.Cols}

	for j := 0; j < mat.Cols; j++ {
		row.Coef[0][j] = mat.Coef[i][j]
	}

	return row
}

func (mat *ComplexMatrix) Tranpose() *ComplexMatrix {
	result := &ComplexMatrix{Rows: mat.Cols, Cols: mat.Rows}

	for i := 0; i < mat.Rows; i++ {
		for j := 0; j < mat.Cols; j++ {
			result.Coef[j][i] = mat.Coef[i][j]
		}
	}

	return result
}

func (mat *ComplexMatrix) H() *ComplexMatrix {
	result := &ComplexMatrix{Rows: mat.Cols, Cols: mat.Rows}

	for i := 0; i < mat.Rows; i++ {
		for j := 0; j < mat.Cols; j++ {
			result.Coef[j][i] = cmplx.Conj(mat.Coef[i][j])
		}
	}

	return result
}

func (mat *ComplexMatrix) Multiply(other *ComplexMatrix) (*ComplexMatrix, error) {
	if mat.Cols != other.Rows {
		return nil, errors.New("incompatible matrices")
	}

	result := &ComplexMatrix{Rows: mat.Rows, Cols: other.Cols}

	for i := 0; i < mat.Rows; i++ {
		for j := 0; j < other.Cols; j++ {
			for k := 0; k < other.Rows; k++ {
				result.Coef[i][j] += mat.Coef[i][k] * other.Coef[k][j]
			}
		}
	}

	return result, nil
}

func (first *ComplexMatrix) Dot(second *ComplexMatrix) complex128 {
	value := complex(0, 0)

	for i := 0; i < first.Rows; i++ {
		value += first.Coef[i][0] * second.Coef[i][0]
	}

	return value
}

func (mat *ComplexMatrix) SubMatrix(rowStart, colStart, rowSize, colSize int) *ComplexMatrix {
	sub := &ComplexMatrix{Rows: rowSize, Cols: colSize}

	for i := 0; i < rowSize; i++ {
		sub.Coef[i] = mat.Coef[i+rowStart][colStart : colStart+colSize]
	}

	return sub
}

func (mat *ComplexMatrix) Add(other *ComplexMatrix) (*ComplexMatrix, error) {
	if mat.Cols != other.Cols || mat.Rows != other.Rows {
		return nil, errors.New("incompatible matrices")
	}

	result := &ComplexMatrix{Rows: mat.Rows, Cols: other.Cols}

	for i := 0; i < mat.Rows; i++ {
		for j := 0; j < mat.Cols; j++ {
			result.Coef[i][j] = mat.Coef[i][j] + other.Coef[i][j]
		}
	}

	return result, nil
}

func (mat *ComplexMatrix) Subtract(other *ComplexMatrix) (*ComplexMatrix, error) {
	if mat.Cols != other.Cols || mat.Rows != other.Rows {
		return nil, errors.New("incompatible matrices")
	}

	result := &ComplexMatrix{Rows: mat.Rows, Cols: other.Cols}

	for i := 0; i < mat.Rows; i++ {
		for j := 0; j < mat.Cols; j++ {
			result.Coef[i][j] = mat.Coef[i][j] - other.Coef[i][j]
		}
	}

	return result, nil
}

func (mat *ComplexMatrix) MultiplyByScalar(scalar float64) (*ComplexMatrix, error) {
	result := &ComplexMatrix{Rows: mat.Rows, Cols: mat.Cols}

	for i := 0; i < mat.Rows; i++ {
		for j := 0; j < mat.Cols; j++ {
			result.Coef[i][j] = mat.Coef[i][j] * complex(scalar, 0)
		}
	}

	return result, nil
}

func (mat *ComplexMatrix) DivideByScalar(scalar float64) (*ComplexMatrix, error) {
	if scalar == 0.0 {
		return nil, errors.New("cannot divide by zero")
	}

	result := &ComplexMatrix{Rows: mat.Rows, Cols: mat.Cols}

	for i := 0; i < mat.Rows; i++ {
		for j := 0; j < mat.Cols; j++ {
			result.Coef[i][j] = mat.Coef[i][j] / complex(scalar, 0)
		}
	}

	return result, nil
}

// func (mat *Matrix) ApplyGaussianElimination(rhs *Matrix) {
// 	for i := 0; i < mat.Cols; i++ {
// 		pivot := mat.Coef[i][i]

// 		for k := i + 1; k < mat.Rows; k++ {
// 			f := mat.Coef[k][i] / pivot

// 			for j := i; j < mat.Cols; j++ {
// 				mat.Coef[k][j] = mat.Coef[k][j] - mat.Coef[i][j]*f
// 			}

// 			rhs.Coef[k][0] = rhs.Coef[k][0] - rhs.Coef[i][0]*f
// 		}
// 	}
// }

// func (mat *Matrix) ApplyLUDecomposition() (*Matrix, *Matrix) {
// 	L := Eye(mat.Rows, mat.Cols)
// 	U := Zeros(mat.Rows, mat.Cols)

// 	err := Copy(U, mat)

// 	if err != nil {
// 		return nil, nil
// 	}

// 	for i := 0; i < mat.Cols; i++ {
// 		pivot := mat.Coef[i][i]

// 		for k := i + 1; k < mat.Rows; k++ {
// 			f := mat.Coef[k][i] / pivot

// 			L.Coef[k][i] = f
// 			for j := i; j < mat.Cols; j++ {
// 				U.Coef[k][j] = U.Coef[k][j] - U.Coef[i][j]*f
// 			}
// 		}
// 	}

// 	return L, U
// }

// func (mat *Matrix) ApplyPartialPivoting() {
// 	for i := 0; i < mat.Cols; i++ {
// 		mi := i
// 		for l := i + 1; l < mat.Rows; l++ {
// 			if mat.Coef[l][i] > mat.Coef[i][i] {
// 				mi = l
// 			}
// 		}

// 		if mi != i {
// 			row := mat.Coef[i]
// 			mat.Coef[i] = mat.Coef[mi]
// 			mat.Coef[mi] = row
// 		}

// 		pivot := mat.Coef[i][i]

// 		for k := i + 1; k < mat.Rows; k++ {
// 			f := mat.Coef[k][i] / pivot

// 			for j := i; j < mat.Cols; j++ {
// 				mat.Coef[k][j] = mat.Coef[k][j] - mat.Coef[i][j]*f
// 			}
// 		}
// 	}
// }

// func (matrix *Matrix) Inverse() (*Matrix, error) {
// 	mat := Zeros(matrix.Rows, matrix.Cols)
// 	err := Copy(mat, matrix)

// 	if err != nil {
// 		return nil, errors.New("couldn't copy matrix")
// 	}

// 	inverse := Eye(mat.Rows, mat.Cols)

// 	for i := 0; i < mat.Cols; i++ {
// 		mi := i
// 		for l := i + 1; l < mat.Rows; l++ {
// 			if mat.Coef[l][i] > mat.Coef[i][i] {
// 				mi = l
// 			}
// 		}

// 		if mi != i {
// 			row := mat.Coef[i]
// 			mat.Coef[i] = mat.Coef[mi]
// 			mat.Coef[mi] = row

// 			row = inverse.Coef[i]
// 			inverse.Coef[i] = inverse.Coef[mi]
// 			inverse.Coef[mi] = row
// 		}

// 		pivot := mat.Coef[i][i]

// 		for k := i + 1; k < mat.Rows; k++ {
// 			f := mat.Coef[k][i] / pivot

// 			for j := 0; j < mat.Cols; j++ {
// 				mat.Coef[k][j] = mat.Coef[k][j] - mat.Coef[i][j]*f
// 				inverse.Coef[k][j] = inverse.Coef[k][j] - inverse.Coef[i][j]*f
// 			}
// 		}
// 	}

// 	for i := 0; i < mat.Rows; i++ {
// 		divisor := mat.Coef[i][i]

// 		for l := 0; l < mat.Cols; l++ {
// 			mat.Coef[i][l] = mat.Coef[i][l] / divisor
// 			inverse.Coef[i][l] = inverse.Coef[i][l] / divisor
// 		}
// 	}

// 	for i := 0; i < mat.Rows; i++ {
// 		for l := i + 1; l < mat.Cols; l++ {
// 			value := mat.Coef[i][l]

// 			for k := 0; k < mat.Cols; k++ {
// 				mat.Coef[i][k] = mat.Coef[i][k] - mat.Coef[l][k]*value
// 				inverse.Coef[i][k] = inverse.Coef[i][k] - inverse.Coef[l][k]*value
// 			}
// 		}
// 	}

// 	return inverse, nil
// }

// func BackSubstitution(upperTriangular *Matrix, rhs *Matrix) *Matrix {
// 	solutions := Zeros(upperTriangular.Cols, 1)

// 	for i := upperTriangular.Rows - 1; i >= 0; i-- {
// 		solutions.Coef[i][0] = rhs.Coef[i][0]

// 		for j := i + 1; j < upperTriangular.Cols; j++ {
// 			solutions.Coef[i][0] = solutions.Coef[i][0] - upperTriangular.Coef[i][j]*solutions.Coef[j][0]
// 		}

// 		solutions.Coef[i][0] = solutions.Coef[i][0] / upperTriangular.Coef[i][i]
// 	}

// 	return solutions
// }

// func ForwardSubstitution(lowerTriangular *Matrix, rhs *Matrix) *Matrix {
// 	solutions := Zeros(lowerTriangular.Cols, 1)

// 	for i := 0; i < rhs.Rows; i++ {
// 		solutions.Coef[i][0] = rhs.Coef[i][0]

// 		for j := 0; j < i; j++ {
// 			solutions.Coef[i][0] = solutions.Coef[i][0] - lowerTriangular.Coef[i][j]*solutions.Coef[j][0]
// 		}

// 		solutions.Coef[i][0] = solutions.Coef[i][0] / lowerTriangular.Coef[i][i]
// 	}

// 	return solutions
// }

// func LeastSquares(A *Matrix, b *Matrix) *Matrix {
// 	At := A.Tranpose()

// 	AtA, _ := At.Multiply(A)
// 	Atb, _ := At.Multiply(b)

// 	AtA.ApplyGaussianElimination(Atb)

// 	x_hat := BackSubstitution(AtA, Atb)

// 	return x_hat
// }

// // Compute the Householder matrix
// // H = I - 2uuᵀ/uᵀu -> Hx = v (reflected)
// func Householder(vector *Matrix) *Matrix {
// 	v_norm := math.Sqrt(Dot(vector, vector))

// 	e_1 := Zeros(vector.Rows, 1)
// 	e_1.Coef[0][0] = 1.0

// 	v_norm_e1, _ := e_1.MultiplyByScalar(v_norm)

// 	u, _ := vector.Subtract(v_norm_e1)

// 	u_norm := math.Sqrt(Dot(u, u))

// 	v_1, _ := u.MultiplyByScalar(1 / u_norm)

// 	v1_v1_t, _ := v_1.Multiply(v_1.Tranpose())

// 	v1_v1_t_2, _ := v1_v1_t.MultiplyByScalar(2.0)

// 	H, _ := Eye(vector.Rows, vector.Rows).Subtract(v1_v1_t_2)

// 	return H
// }

// func Diag(first *Matrix, second *Matrix) *Matrix {
// 	m := first.Rows + second.Rows
// 	n := first.Cols + second.Cols

// 	mat := Zeros(m, n)

// 	for i := 0; i < first.Rows; i++ {
// 		for j := 0; j < first.Cols; j++ {
// 			mat.Coef[i][j] = first.Coef[i][j]
// 		}
// 	}

// 	for i := first.Rows; i < m; i++ {
// 		for j := first.Cols; j < n; j++ {
// 			mat.Coef[i][j] = second.Coef[i-first.Rows][j-first.Cols]
// 		}
// 	}

// 	return mat
// }

// func QRFactorization(mat *Matrix) (*Matrix, *Matrix) {
// 	m := mat.Rows
// 	n := mat.Cols

// 	p := min(m, n)

// 	R := mat
// 	H := Eye(m, m)

// 	for i := 0; i < p; i++ {
// 		//Get the bottom right matrix
// 		bottom_right := R.SubMatrix(i, i, m-i, n-i)

// 		H_bar := Householder(bottom_right.Col(0))
// 		I_i := Eye(i, i)

// 		H_i := Diag(I_i, H_bar)

// 		R, _ = H_i.Multiply(R)

// 		H, _ = H_i.Multiply(H)
// 	}

// 	Q := H.Tranpose()

// 	return Q, R
// }

// func PowerMethod(mat *Matrix, n int) (*Matrix, float64) {
// 	q_0 := Ones(mat.Rows, 1)

// 	q_k := q_0

// 	var lambda_k float64

// 	for i := 0; i < n; i++ {
// 		z_k, _ := mat.Multiply(q_k)

// 		lambda_k = Max(z_k)
// 		q_k, _ = z_k.MultiplyByScalar(1 / lambda_k)
// 	}

// 	return q_k, lambda_k
// }

// func InversePowerMethod(mat *Matrix, n int) (*Matrix, float64) {
// 	q_0 := Ones(mat.Rows, 1)

// 	q_k := q_0

// 	var lambda_k float64

// 	L, U := mat.ApplyLUDecomposition()

// 	for i := 0; i < n; i++ {
// 		//z_k, _ := mat.Multiply(q_k)
// 		temp_k := ForwardSubstitution(L, q_k)
// 		q_k = BackSubstitution(U, temp_k)

// 		lambda_k = Max(q_k)

// 		q_k, _ = q_k.MultiplyByScalar(1 / lambda_k)

// 		//q_k, _ = q_k.MultiplyByScalar(1 / q_k.Norm())
// 	}

// 	return q_k, lambda_k
// }
