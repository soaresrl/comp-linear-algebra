package main

import (
	"fmt"

	"github.com/soaresrl/linalg/pkg/matrix"
)

func main() {
	// arr := [][]float64{
	// 	{1, 4, 7},
	// 	{2, 5, 8},
	// 	{3, 6, 10},
	// }
	arr := [][]float64{
		{-1, -1, 1},
		{1, 3, 3},
		{-1, -1, 5},
	}

	B := matrix.FromArray(arr)

	Q, R := matrix.QRFactorization(B)
	// x1_vec := B.Col(0)

	// hh1 := matrix.Householder(x1_vec)

	fmt.Printf("%v\n", *Q)
	fmt.Printf("%v\n", *R)

	// R1, _ := hh1.Multiply(B)

	//fmt.Printf("%v\n", *R1)
}
