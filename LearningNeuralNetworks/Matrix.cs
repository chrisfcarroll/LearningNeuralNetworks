using System;
using System.Diagnostics.Contracts;

namespace LearningNeuralNetworks
{
    public class MatrixF
    {
        public double this[int rowIndex, int columnIndex] { get { return data[rowIndex, columnIndex]; } set { data[rowIndex, columnIndex] = value; } }
        public int ColumnCount { get; }
        public int RowCount { get; }

        public MatrixF(int rowCount, int columnCount)
        {
            RowCount = rowCount;
            ColumnCount = columnCount;
            data = new double[rowCount, columnCount];
        }

        public MatrixF(double[,] dataSourceByReference)
        {
            data = dataSourceByReference;
            RowCount = data.GetLength(0);
            ColumnCount = data.GetLength(1);
        }

        readonly double[,] data;

        public MatrixF Copy()
        {
            var clone= new MatrixF(RowCount,ColumnCount);
            Array.Copy(data, clone.data, ColumnCount*RowCount);
            return clone;
        }

        public static MatrixF operator -(MatrixF value)
        {
            var result = value.Copy();
            for (int i = 0; i < result.RowCount ; i++)
            for (int j = 0; j < result.ColumnCount; j++)
            {
                result[i, j] = -result[i, j];
            }
            return result;
        }

        public static MatrixF operator +(MatrixF left, MatrixF right)
        {
            Contract.Requires(
                left.ColumnCount==right.ColumnCount && left.RowCount==right.RowCount,
                $"The matrices must be of the same size and shape but left is {left.RowCount},{left.ColumnCount} and right is {right.RowCount},{right.ColumnCount}");

            var result = left.Copy();
            for (int i = 0; i < result.RowCount; i++)
            for (int j = 0; j < result.ColumnCount; j++)
            {
                result[i, j] = result[i, j] + right[i,j];
            }
            return result;
        }
        public static MatrixF operator -(MatrixF left, MatrixF right)
        {
            Contract.Requires(
                left.ColumnCount == right.ColumnCount && left.RowCount == right.RowCount,
                $"The matrices must be of the same size and shape but left is {left.RowCount},{left.ColumnCount} and right is {right.RowCount},{right.ColumnCount}");

            var result = left.Copy();
            for (int i = 0; i < result.RowCount; i++)
                for (int j = 0; j < result.ColumnCount; j++)
                {
                    result[i, j] = result[i, j] - right[i, j];
                }
            return result;
        }
        public static MatrixF operator *(MatrixF value, double scalar)
        {
            var result = value.Copy();
            for (int i = 0; i < result.RowCount; i++)
                for (int j = 0; j < result.ColumnCount; j++)
                {
                    result[i, j] = result[i, j] * scalar;
                }
            return result;
        }
        public static MatrixF operator *(double scalar, MatrixF value)
        {
            var result = value.Copy();
            for (int i = 0; i < result.RowCount; i++)
                for (int j = 0; j < result.ColumnCount; j++)
                {
                    result[i, j] = result[i, j] * scalar;
                }
            return result;
        }
    }
}
