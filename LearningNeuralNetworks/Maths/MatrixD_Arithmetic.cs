using System;
using System.Linq;
using LearningNeuralNetworks.Frameworks;

namespace LearningNeuralNetworks.Maths
{
    /// <summary>A Matrix of System.Double</summary>
    public partial class MatrixD
    {
        public double this[int rowIndex, int columnIndex] { get { return data[rowIndex][columnIndex]; } set { data[rowIndex][columnIndex] = value; } }
        public int ColumnCount { get; }
        public int RowCount { get; }
        readonly double[][] data;

        public static MatrixD dot(MatrixD left, MatrixD right) { return left * right; }
        public MatrixD dot(MatrixD right) { return this * right; }

        public static MatrixD operator *(MatrixD left, MatrixD right)
        {
            (left.ColumnCount== right.RowCount).ElseThrow(new ArgumentOutOfRangeException(nameof(right), $"Can't multiply a [{left.RowCount},{left.ColumnCount}] matrix by a [{right.RowCount},{right.ColumnCount}]"));
            //
            var result= new MatrixD(left.RowCount, right.ColumnCount);
            for (int i = 0; i < result.RowCount; i++)
            for (int j = 0; j < result.ColumnCount; j++)
            for (int k = 0; k < left.ColumnCount; k++)
            {
                result[i, j] += left[i, k]*right[k, j];
            }
            return result;
        }

        public static MatrixD operator -(MatrixD value)
        {
            var result = value.Copy();
            for (int i = 0; i < result.RowCount ; i++)
            for (int j = 0; j < result.ColumnCount; j++)
            {
                result[i, j] = -result[i, j];
            }
            return result;
        }

        public static MatrixD operator +(MatrixD left, MatrixD right)
        {
            (left.ColumnCount==right.ColumnCount && left.RowCount==right.RowCount).ElseThrow(new ArgumentOutOfRangeException(nameof(right),$"The matrices must be of the same size and shape but left is {left.RowCount},{left.ColumnCount} and right is {right.RowCount},{right.ColumnCount}"));
            //
            var result = left.Copy();
            for (int i = 0; i < result.RowCount; i++)
            for (int j = 0; j < result.ColumnCount; j++)
            {
                result[i, j] = result[i, j] + right[i,j];
            }
            return result;
        }
        public static MatrixD operator -(MatrixD left, MatrixD right)
        {
            (left.ColumnCount == right.ColumnCount && left.RowCount == right.RowCount).ElseThrow(new ArgumentOutOfRangeException(nameof(right),$"The matrices must be of the same size and shape but left is {left.RowCount},{left.ColumnCount} and right is {right.RowCount},{right.ColumnCount}"));

            var result = left.Copy();
            for (int i = 0; i < result.RowCount; i++)
            for (int j = 0; j < result.ColumnCount; j++)
            {
                result[i, j] = result[i, j] - right[i, j];
            }
            return result;
        }
        public static MatrixD operator *(MatrixD value, double scalar) { return value.data.Select(r => r.Select(el => el * scalar).ToArray()).ToArray(); }

        public static MatrixD operator *(double scalar, MatrixD value) { return value * scalar;}

        public static MatrixD operator +(MatrixD value, double scalar) { return value.data.Select(r => r.Select(el => el + scalar).ToArray()).ToArray(); }

        public static MatrixD operator +(double scalar, MatrixD value) { return value + scalar; }

        public static MatrixD operator -(MatrixD value, double scalar) { return value + -scalar; }


        ///<summary>Matrix Transpose</summary>
        /// <returns>The transpose of this matrix</returns>
        public MatrixD T()
        {
            var transpose = new MatrixD(ColumnCount, RowCount);
            for(int i=0; i<RowCount;i++)for(int j=0; j<ColumnCount; j++)
            {
                transpose[j, i] = this[i, j];
            }
            return transpose;
        }
    }
}
