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
	// arr := [][]float64{
	// 	{-1, -1, 1},
	// 	{1, 3, 3},
	// 	{-1, -1, 5},
	// }

	arr := [][]float64{
		{2, 1},
		{2, 3},
	}

	B := matrix.FromArray(arr)

	//Q, R := matrix.QRFactorization(B)
	// x1_vec := B.Col(0)
	fmt.Printf("%v\n", *B)

	// hh1 := matrix.Householder(x1_vec)

	q, lambda := matrix.PowerMethod(B, 10)

	inv_q, inv_lambda := matrix.InversePowerMethod(B, 10)
	fmt.Printf("%v\n", *B)

	fmt.Printf("%v\n", *q)
	fmt.Printf("%v\n", lambda)

	fmt.Printf("%v\n", *inv_q)
	fmt.Printf("%v\n", inv_lambda)
	// fmt.Printf("%v\n", *R)

	// R1, _ := hh1.Multiply(B)

	//fmt.Printf("%v\n", *R1)
}
