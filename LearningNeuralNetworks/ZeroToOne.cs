using System;
using System.Collections.Generic;
using System.Linq;

namespace LearningNeuralNetworks
{
    /// <summary>
    /// Models a double value in the range 0 &lt;= value &lt;= 1
    /// </summary>
    public struct ZeroToOne : IEquatable<ZeroToOne>
    {
        /// <summary>Two values which differ by less than this will be considered equal by <see cref="Equals(LearningNeuralNetworks.ZeroToOne)"/></summary>
        public const double Epsilon = 1e-15;
        readonly double value;

        public ZeroToOne(double input)
        {
            if(input < 0 || input > 1) throw new ArgumentOutOfRangeException(nameof(input),"Must be between 0d and +1d");
            value = input;
        }

        public ZeroToOne(bool input) { value = input ? 1 : 0; }

        ///<returns>True if and only if this value > 0.5d</returns>
        public bool AsBool => value > 0.5d;

        public static implicit operator double(ZeroToOne input) { return input.value;}
        public static implicit operator ZeroToOne(double input) { return new ZeroToOne(input);}
        public static implicit operator bool(ZeroToOne input) { return input.AsBool;}

        public static implicit operator ZeroToOne(bool input) { return new ZeroToOne(input?1:0);}

        /// <summary>
        /// Returns <see cref="bool.True"/> iff the values differ by less than <see cref="Epsilon"/>
        /// </summary>
        public bool Equals(ZeroToOne other) { return Math.Abs(value - other.value) < Epsilon; }

        public override bool Equals(object obj) 
        {
            if (ReferenceEquals(null, obj)) return false;
            return obj is ZeroToOne && Equals((ZeroToOne) obj);
        }

        public override int GetHashCode() { return value.GetHashCode(); }

        public static bool operator ==(ZeroToOne left, ZeroToOne right) { return left.Equals(right); }

        public static bool operator !=(ZeroToOne left, ZeroToOne right) { return !left.Equals(right); }

        public override string ToString() { return value.ToString(); }
        public string ToString(string format) { return value.ToString(format); }
        public string ToString(IFormatProvider formatter) { return value.ToString(formatter); }
    }

    public static class ZeroToOneExtensions
    {
        public static IEnumerable<ZeroToOne> ToZeroToOnes(this double[] inputs) { return inputs.Select(x => (ZeroToOne)x); }
        public static ZeroToOne[] ToZeroToOneArray(this double[] inputs) { return ToZeroToOnes(inputs).ToArray(); }
    }
}