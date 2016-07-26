using System;
using System.Diagnostics.Contracts;
using System.Text;

namespace LearningNeuralNetworks.Maths
{
    /// <summary>A Matrix of System.Double</summary>
    public class MatrixD : IEquatable<MatrixD>
    {
        public double this[int rowIndex, int columnIndex] { get { return data[rowIndex, columnIndex]; } set { data[rowIndex, columnIndex] = value; } }
        public int ColumnCount { get; }
        public int RowCount { get; }

        public MatrixD(int rowCount, int columnCount)
        {
            RowCount = rowCount;
            ColumnCount = columnCount;
            data = new double[rowCount, columnCount];
        }

        public MatrixD(double[,] dataSourceByReference)
        {
            data = dataSourceByReference;
            RowCount = data.GetLength(0);
            ColumnCount = data.GetLength(1);
        }

        readonly double[,] data;
        static readonly double Epsilon= 1e-100;

        public MatrixD Copy()
        {
            var clone= new MatrixD(RowCount,ColumnCount);
            Array.Copy(data, clone.data, ColumnCount*RowCount);
            return clone;
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
        public static MatrixD operator -(MatrixD left, MatrixD right)
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
        public static MatrixD operator *(MatrixD value, double scalar)
        {
            var result = value.Copy();
            for (int i = 0; i < result.RowCount; i++)
            for (int j = 0; j < result.ColumnCount; j++)
            {
                result[i, j] = result[i, j] * scalar;
            }
            return result;
        }
        public static MatrixD operator *(double scalar, MatrixD value)
        {
            var result = value.Copy();
            for (int i = 0; i < result.RowCount; i++)
                for (int j = 0; j < result.ColumnCount; j++)
                {
                    result[i, j] = result[i, j] * scalar;
                }
            return result;
        }

        public bool Equals(MatrixD other)
        {
            if (ReferenceEquals(null, other)) return false;
            if (ReferenceEquals(this, other)) return true;
            if (ColumnCount != other.ColumnCount) return false;
            if (RowCount != other.RowCount) return false;
            for(int i=0;   i < RowCount; i++)
            for(int j = 0; j < ColumnCount; j++)
            {
                if (Math.Abs(data[i, j] - other.data[i, j]) > Epsilon) return false;
            }
            return true;
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((MatrixD) obj);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                var hashCode = data.GetHashCode();
                hashCode = (hashCode*397) ^ ColumnCount;
                hashCode = (hashCode*397) ^ RowCount;
                return hashCode;
            }
        }

        public static bool operator ==(MatrixD left, MatrixD right)
        {
            return Equals(left, right);
        }

        public static bool operator !=(MatrixD left, MatrixD right)
        {
            return !Equals(left, right);
        }

        public double[][] ByRows()
        {
            var result = new double[RowCount][];
            for (int i = 0; i < RowCount; i++)
            {
                result[i]= new double[ColumnCount];
                for (int j = 0; j < ColumnCount; j++)
                {
                    result[i][j] = data[i, j];
                }
            }
            return result;
        }

        public override string ToString()
        {
            return ToString("G");
        }

        public string ToString(string format)
        {
            var sb= new StringBuilder("[");
            for (int i = 0; i < RowCount; i++)
            {
                sb.Append('[');
                for (int j = 0; j < ColumnCount; j++)
                {
                    sb.Append(data[i, j].ToString(format)).Append(',').Append(' ');
                }
                sb.Append(']').AppendLine();
            }
            return sb.Append(']').AppendLine().ToString();
        }

        public string ToString(int maxRowsToShow, int maxColumnsToShow, string format="G")
        {
            maxRowsToShow = Math.Min(RowCount, maxRowsToShow);
            maxColumnsToShow = Math.Min(ColumnCount, maxColumnsToShow);
            var halfRows = Math.Max(1, (1 + maxRowsToShow)/2);
            var halfCols = Math.Max(1, (1 + maxColumnsToShow)/2);
            var isElidingCols = ColumnCount > 2*maxColumnsToShow;
            var isElidingRows = RowCount > 2*maxRowsToShow;
            var topRows = new {from = 0, to = halfRows};
            var bottomRows = new { from = RowCount-halfRows, to = RowCount };
            //
            var sb = new StringBuilder("[");
            foreach (var range in new[] {topRows, bottomRows})
            {
                for (int i = range.from; i < range.to; i++)
                {
                    sb.Append('[');
                    for (int j = 0; j < halfCols; j++)
                    {
                        sb.Append(data[i, j].ToString(format)).Append(',').Append(' ');
                    }
                    if (isElidingCols) { sb.Append('…'); }
                    for (int j = ColumnCount - halfCols; j < ColumnCount; j++)
                    {
                        sb.Append(data[i, j].ToString(format)).Append(',').Append(' ');
                    }
                    sb.Append(']').AppendLine();
                }
                if (isElidingRows && range==topRows) for(int k=0; k<maxColumnsToShow;k++){ sb.Append(" … "); }
            }
            return sb.Append(']').AppendLine().ToString();
        }
    }
}
