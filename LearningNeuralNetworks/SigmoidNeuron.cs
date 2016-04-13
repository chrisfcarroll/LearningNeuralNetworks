using System;
using System.Collections.Generic;
using System.Globalization;

namespace LearningNeuralNetworks
{
    /// <summary>
    /// Models a Perceptron, see e.g. <see href="http://neuralnetworksanddeeplearning.com/chap1.html#perceptrons">this introduction</see>. 
    /// With integers, not Reals
    /// </summary>
    public struct SigmoidNeuron
    {
        public const double High = 1e100d;
        public const double Off = 0d;

        public double bias;
        public IList<Sinput> Inputs;

        public ZeroToOne FiringRate => (Inputs.DotProduct() + bias).Sigmoid();
    }

    public struct Sinput
    {
        public double weight;
        public SigmoidNeuron Source;
        public Sinput(SigmoidNeuron source, double weight) : this()
        {
            this.weight = weight;
            this.Source = source;
        }
    }

    public struct ZeroToOne
    {
        readonly double value;

        public ZeroToOne(double input)
        {
            if(input < 0 || input > 1) throw new ArgumentOutOfRangeException(nameof(input),"Must be between 0d and +1d");
            value = input;
        }

        public static implicit operator double(ZeroToOne input) { return input.value;}
        public static implicit operator ZeroToOne(double input) { return new ZeroToOne(input);}

        public override string ToString() { return value.ToString(); }
    }
}