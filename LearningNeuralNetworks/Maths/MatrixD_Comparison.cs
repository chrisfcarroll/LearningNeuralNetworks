using System;

namespace LearningNeuralNetworks.Maths
{
    /// <summary>A Matrix of System.Double</summary>
    public partial class MatrixD : IEquatable<MatrixD>
    {
        static readonly double Epsilon = 1e-100;

        public bool Equals(MatrixD other)
        {
            if (ReferenceEquals(null, other)) return false;
            if (ReferenceEquals(this, other)) return true;
            if (ColumnCount != other.ColumnCount) return false;
            if (RowCount != other.RowCount) return false;
            for (int i = 0; i < RowCount; i++)
            for (int j = 0; j < ColumnCount; j++)
            {
                if (Math.Abs(data[i][j] - other.data[i][j]) > Epsilon) return false;
            }
            return true;
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((MatrixD)obj);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                var hashCode = data.GetHashCode();
                hashCode = (hashCode * 397) ^ ColumnCount;
                hashCode = (hashCode * 397) ^ RowCount;
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
    }
}
