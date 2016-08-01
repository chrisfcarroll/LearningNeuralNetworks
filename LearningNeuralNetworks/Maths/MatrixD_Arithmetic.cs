using System;
using System.Linq;
using System.Runtime.Remoting.Messaging;
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

        /// <summary>
        /// Dot product is regular matrix multiplication <see cref="*"/> for 2-D matrices, 
        /// BUT regular vector dot product for column vectors, i.e. where both operands are matrices with 1 column and the same number of rows and 
        /// </summary>
        /// <param name="right"></param>
        /// <returns></returns>
        public MatrixD dot(MatrixD right)
        {
            if(ColumnCount==right.RowCount) return this * right;
            if (RowCount == right.RowCount && ColumnCount==1 && right.ColumnCount==1) return this.T()*right;
            //
            throw new ArgumentOutOfRangeException(nameof(right), $"Can't dot a [{RowCount},{ColumnCount}] matrix with a [{right.RowCount},{right.ColumnCount}]");
        }

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
            EnsureSameShapeElseThrow(left, right);
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
            EnsureSameShapeElseThrow(left, right);

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

        /// <summary>
        /// Zip operations, or element operations, take 2 matrices of identical shape and apply the operator element-wise.
        /// eg [[1,2,3], [4,5,6]].ZipTimes([[1,1,1],[2,2,2]) is [[1,2,3], [8,10,12]]
        /// </summary>
        /// <param name="right"></param>
        /// <returns>A <see cref="MatrixD"/> of the same shape as the inputs with each element being the product of the input elements</returns>
        public MatrixD ZipTimes(MatrixD right)
        {
            EnsureSameShapeElseThrow(this, right);
            //
            return new MatrixD(data.Zip(right.data, (l,r)=> l.Zip(r, (ll,rr)=> ll*rr).ToArray()).ToArray());
        }


        static void EnsureSameShapeElseThrow(MatrixD left, MatrixD right)
        {
            if (left.ColumnCount != right.ColumnCount || left.RowCount != right.RowCount)
                throw 
                    new ArgumentOutOfRangeException(
                        nameof(right),
                        $"The matrices must be of the same size and shape but left is {left.RowCount},{left.ColumnCount} and right is {right.RowCount},{right.ColumnCount}");
        }
    }
}
