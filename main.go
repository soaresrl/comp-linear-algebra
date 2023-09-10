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
		{3, 17, 10},
		{2, 4, -2},
		{6, 18, -12},
	}

	mat := matrix.FromArray(arr)

	fmt.Printf("%v\n", *mat)

	//L, U := mat.ApplyLUDecomposition()
	//mat.ApplyPartialPivoting()

	inverse, _ := mat.Inverse()

	fmt.Printf("%v\n", *inverse)
}
