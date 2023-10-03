package matrix

import (
	"errors"
)

type Matrix struct {
	Coef	[][]float64
	Rows	int
	Cols	int
}

func FromArray(array [][]float64) *Matrix {
	result := &Matrix{ Coef: array, Rows: len(array), Cols: len(array[0])}

	return result
}

func Zeros(rows int, cols int) *Matrix {
	result := make([][]float64, rows)
	
	for i := range result {
		result[i] = make([]float64, cols)
	}

	mat := &Matrix{ Coef: result, Rows: rows, Cols: cols }

	return mat
}

func Eye(rows int, cols int) *Matrix {
	result := make([][]float64, rows)
	
	for i := range result {
		result[i] = make([]float64, cols)

		result[i][i] = 1.0
	}

	mat := &Matrix{ Coef: result, Rows: rows, Cols: cols }

	return mat
}

func (mat *Matrix) Tranpose() *Matrix {
	result := Zeros(mat.Cols, mat.Rows)
	
	for i := 0; i < mat.Rows; i++ {
		for j := 0; j < mat.Cols; j++ {
			result.Coef[j][i] = mat.Coef[i][j]
		}
	}

	return result
}

func (mat *Matrix) Multiply(other *Matrix) (*Matrix, error)  {
	if mat.Cols != other.Rows {
		return nil, errors.New("Incompatible matrices.")
	}

	result := Zeros(mat.Rows, other.Cols)

	for i := 0; i < mat.Rows; i++ {
		for j := 0; j < other.Cols; j++ {
			for k := 0; k < other.Rows; k++ {
				result.Coef[i][j] += mat.Coef[i][k] * other.Coef[k][j]
			}
		}
	}

	return result, nil
}

func (mat *Matrix) Add(other *Matrix) (*Matrix, error)  {
	if mat.Cols != other.Cols || mat.Rows != other.Rows {
		return nil, errors.New("Incompatible matrices")
	}

	result := Zeros(mat.Rows, other.Cols)

	for i := 0; i < mat.Rows; i++ {
		for j := 0; j < mat.Cols; j++ {
			result.Coef[i][j] = mat.Coef[i][j] + other.Coef[i][j]
		}
	}

	return result, nil
}


func (mat *Matrix) Subtract(other *Matrix) (*Matrix, error)  {
	if mat.Cols != other.Cols || mat.Rows != other.Rows {
		return nil, errors.New("Incompatible matrices")
	}

	result := Zeros(mat.Rows, other.Cols)

	for i := 0; i < mat.Rows; i++ {
		for j := 0; j < mat.Cols; j++ {
			result.Coef[i][j] = mat.Coef[i][j] - other.Coef[i][j]
		}
	}

	return result, nil
}

func (mat *Matrix) MultiplyByScalar(scalar float64) (*Matrix, error)  {
	result := Zeros(mat.Rows, mat.Cols)

	for i := 0; i < mat.Rows; i++ {
		for j := 0; j < mat.Cols; j++ {
			result.Coef[i][j] = mat.Coef[i][j] * scalar
		}
	}

	return result, nil
}

func (mat *Matrix) DivideByScalar(scalar float64) (*Matrix, error)  {
	if scalar == 0.0 {
		return nil, errors.New("Cannot divide by zero.")
	}

	result := Zeros(mat.Rows, mat.Cols)

	for i := 0; i < mat.Rows; i++ {
		for j := 0; j < mat.Cols; j++ {
			result.Coef[i][j] = mat.Coef[i][j] / scalar
		}
	}

	return result, nil
}

func (mat *Matrix) ApplyGaussianElimination(rhs *Matrix) {
	for i := 0; i < mat.Cols; i++ {
		pivot := mat.Coef[i][i]

		for k := i + 1; k < mat.Rows; k ++ {
			f := mat.Coef[k][i] / pivot

			for j := i; j < mat.Cols; j++ {
				mat.Coef[k][j] = mat.Coef[k][j] - mat.Coef[i][j] * f
			}

			rhs.Coef[k][0] = rhs.Coef[k][0] - rhs.Coef[i][0] * f
		}
	}
}

func (mat *Matrix) ApplyLUDecomposition() (*Matrix, *Matrix) {
	L := Eye(mat.Rows, mat.Cols)
	U := *mat

	for i := 0; i < mat.Cols; i++ {
		pivot := mat.Coef[i][i]

		for k := i + 1; k < mat.Rows; k ++ {
			f := mat.Coef[k][i] / pivot

			L.Coef[k][i] = f
			for j := i; j < mat.Cols; j++ {
				U.Coef[k][j] = U.Coef[k][j] - U.Coef[i][j] * f
			}
		}
	}

	return L, &U
}

func (mat *Matrix) ApplyPartialPivoting() {
	for i := 0; i < mat.Cols; i++ {
		mi := i
		for l := i + 1; l < mat.Rows; l++ {
			if mat.Coef[l][i] > mat.Coef[i][i] {
				mi = l
			}
		}

		if mi != i {
			row := mat.Coef[i]
			mat.Coef[i] = mat.Coef[mi]
			mat.Coef[mi] = row
		}

		pivot := mat.Coef[i][i]

		for k := i + 1; k < mat.Rows; k ++ {
			f := mat.Coef[k][i] / pivot

			for j := i; j < mat.Cols; j++ {
				mat.Coef[k][j] = mat.Coef[k][j] - mat.Coef[i][j] * f
			}
		}
	}
}

func (mat Matrix) Inverse() (*Matrix, error) {
	inverse := Eye(mat.Rows, mat.Cols)

	for i := 0; i < mat.Cols; i++ {
		mi := i
		for l := i + 1; l < mat.Rows; l++ {
			if mat.Coef[l][i] > mat.Coef[i][i] {
				mi = l
			}
		}

		if mi != i {
			row := mat.Coef[i]
			mat.Coef[i] = mat.Coef[mi]
			mat.Coef[mi] = row

			row = inverse.Coef[i]
			inverse.Coef[i] = inverse.Coef[mi]
			inverse.Coef[mi] = row
		}

		pivot := mat.Coef[i][i]

		for k := i + 1; k < mat.Rows; k ++ {
			f := mat.Coef[k][i] / pivot

			for j := 0; j < mat.Cols; j++ {
				mat.Coef[k][j] = mat.Coef[k][j] - mat.Coef[i][j] * f
				inverse.Coef[k][j] = inverse.Coef[k][j] - inverse.Coef[i][j] * f
			}
		}
	}

	for i := 0; i < mat.Rows; i++ {
		divisor := mat.Coef[i][i];
	
		for l := 0; l < mat.Cols; l++ {
			mat.Coef[i][l] = mat.Coef[i][l] / divisor
			inverse.Coef[i][l] = inverse.Coef[i][l] / divisor
		}
	}

	for i := 0; i < mat.Rows; i++ {
		for l := i + 1; l < mat.Cols; l++ {
			value := mat.Coef[i][l]

			for k := 0; k < mat.Cols; k++ {
				mat.Coef[i][k] = mat.Coef[i][k] - mat.Coef[l][k] * value
				inverse.Coef[i][k] = inverse.Coef[i][k] - inverse.Coef[l][k] * value
			}
		}
	}

	return inverse, nil
}

func BackSubstitution(upperTriangular *Matrix, rhs *Matrix) *Matrix {
	solutions := Zeros(upperTriangular.Cols, 1)

	for i := upperTriangular.Rows - 1; i >= 0; i-- {
		solutions.Coef[i][0] = rhs.Coef[i][0]

		for j := i + 1; j < upperTriangular.Cols; j++ {
			solutions.Coef[i][0] = solutions.Coef[i][0] - upperTriangular.Coef[i][j] * solutions.Coef[j][0]
		}

		solutions.Coef[i][0] = solutions.Coef[i][0] / upperTriangular.Coef[i][i]
	}

	return solutions
}

func LeastSquares(A *Matrix, b *Matrix) *Matrix {
		At := A.Tranpose()

		AtA, _ := At.Multiply(A)
		Atb, _ := At.Multiply(b)

		AtA.ApplyGaussianElimination(Atb)

		x_hat := BackSubstitution(AtA, Atb)

		return x_hat
}