using System.Collections.Generic;
using System.Linq;

namespace LearningNeuralNetworks
{
    static class PerceptronExtensionMethods
    {
        public static int DotProduct(this IEnumerable<Pinput> inputs)
        {
            return inputs==null ?
                    0 
                    : inputs.Sum(i => i.Source.IsFiring ? i.weight : 0);
        }
    }
}