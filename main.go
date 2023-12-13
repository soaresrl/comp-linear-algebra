package main

import (
	"fmt"
	"runtime"

	"github.com/soaresrl/linalg/pkg/matrix"
)

func TestRREF() {
	A := matrix.NewRealMatrix(4, 4)

	arr := [][]float64{
		{1, 2, 4, -1},
		{4, 5, 10, 3},
		{7, 8, 16, 5},
		{7, 8, 16, 5},
	}

	A.FromArray(arr)

	rref, rank, nullDim := A.RREF()

	fmt.Printf("%v\n", *rref)
	fmt.Printf("rank: %d\n", rank)
	fmt.Printf("Dimension of Null(A): %d\n", nullDim)
}

func TestEigen() {
	A := matrix.NewRealMatrix(2, 2)

	// arr := [][]float64{
	// 	{30, 6, 5},
	// 	{6, 30, 9},
	// 	{5, 9, 30},
	// }

	arr := [][]float64{
		{3.0, 2.0},
		{2.0, 3.0},
	}

	// arr := [][]float64{
	// 	{4.0, 3.0},
	// 	{8.0, 6.0},
	// }

	A.FromArray(arr)

	A.SVDDecomposition()

	//A.SimmetricEigenValues()
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())

	//TestRREF()
	TestEigen()
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

	// A := matrix.NewRealMatrix(3, 3)

	// arr := [][]float64{
	// 	{2, -1, 0},
	// 	{-1, 2, -1},
	// 	{0, -1, 2},
	// }

	// A.FromArray(arr)

	// b := matrix.NewRealMatrix(3, 1)

	// arr_b := [][]float64{
	// 	{0},
	// 	{1},
	// 	{2},
	// }

	// b.FromArray(arr_b)

	// arr := [][]float64{
	// 	{1, -2, -3},
	// 	{3, 5, 2},
	// 	{2, 3, -1},
	// }

	// B.FromArray(arr)

	// q := matrix.NewRealMatrix(3, 1)

	// arr_q := [][]float64{
	// 	{0},
	// 	{0},
	// 	{-1},
	// }

	// q.FromArray(arr_q)

	//Q, R := matrix.QRFactorization(B)
	// x1_vec := B.Col(0)
	//fmt.Printf("%v\n", *A)

	// A, _ := B.RREF()

	// sol := B.CompletePivoting(q)

	// x := A.SOR(b)

	// fmt.Printf("%v\n", *x)
	// Q, R := B.QRGramSchmidt()

	// fmt.Printf("%v\n", *Q)
	// fmt.Printf("%v\n", *R)

	// value := matrix.Dot(Q.Col(0), Q.Col(1))
	// value_2 := matrix.Dot(Q.Col(0), Q.Col(2))
	// value_3 := matrix.Dot(Q.Col(1), Q.Col(2))

	// fmt.Println(value)
	// fmt.Println(value_2)
	// fmt.Println(value_3)
	// hh1 := matrix.Householder(x1_vec)

	// q, lambda := matrix.PowerMethod(B, 10)

	// inv_q, inv_lambda := matrix.InversePowerMethod(B, 10)
	// fmt.Printf("%v\n", *B)

	// fmt.Printf("%v\n", *q)
	// fmt.Printf("%v\n", lambda)

	// fmt.Printf("%v\n", *inv_q)
	// fmt.Printf("%v\n", inv_lambda)
	// fmt.Printf("%v\n", *R)

	// R1, _ := hh1.Multiply(B)

	//fmt.Printf("%v\n", *R1)
}
