using System;
using System.Collections.Generic;
using System.Linq;

namespace LearningNeuralNetworks
{
    static class SigmoidNeuronExtensionMethods
    {
        public static double DotProduct(this IEnumerable<Sinput> inputs)
        {
            return inputs == null ?
                0
                : inputs.Sum(i => i.Source.FiringRate * i.weight);
        }

        public static double Sigmoid(this double input)
        {
            return 1d / (1d + Math.Exp(-input));
        }

        public static double SigmoidDerivative(this double input)
        {
            var sigmoid = Sigmoid(input);
            return  sigmoid * (1 - sigmoid);
        }

        public static double Sigmoid(this ZeroToOne input)
        {
            return 1d / (1d + Math.Exp(-input));
        }

        public static double SigmoidDerivative(this ZeroToOne input)
        {
            var sigmoid = Sigmoid(input);
            return sigmoid * (1 - sigmoid);
        }

        public static double Sqrt(this double input)
        {
            return Math.Sqrt(input);
        }

    }
}