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

        public static ZeroToOne Sigmoid(this double input)
        {
            return new ZeroToOne(1d / (1d + Math.Exp(-input)));
        }
    }
}