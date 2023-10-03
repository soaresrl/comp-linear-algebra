package main

import (
	"fmt"
	"github.com/soaresrl/linalg/pkg/matrix"
)

func main(){
	// arr := [][]float64{
	// 	{1, 4, 7},
	// 	{2, 5, 8},
	// 	{3, 6, 10},
	// }
	arr := [][]float64{
		{1, -2},
		{1, 0},
		{1, 2},
	}

	solArr := [][]float64{
		{1},
		{2},
		{4},
	}

	mat := matrix.FromArray(arr)
	solMat := matrix.FromArray(solArr)

	matCopy := matrix.FromArray(arr)
	solMatCopy := matrix.FromArray(solArr)
	

	fmt.Printf("%v\n", *mat)
	fmt.Printf("%v\n", *solMat)

	//L, U := mat.ApplyLUDecomposition()
	//mat.ApplyGaussianElimination(solMat)

	//x := matrix.BackSubstitution(mat, solMat)

	x_hat := matrix.LeastSquares(matCopy, solMatCopy)

	//x := matrix.BackSubstitution(U, y)
	//mat.ApplyPartialPivoting()

	//inverse, _ := mat.Inverse()

	//fmt.Printf("%v\n", *x)
	fmt.Printf("%v\n", *x_hat)
}
