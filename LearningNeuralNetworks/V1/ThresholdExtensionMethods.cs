using System.Collections.Generic;
using System.Linq;

namespace LearningNeuralNetworks.V1
{
    static class ThresholdExtensionMethods
    {
        public static int DotProduct(this IEnumerable<ThresholdNeuron.Input> inputs)
        {
            return inputs==null ?
                    0 
                    : inputs.Sum(i => i.Source.IsFiring ? i.weight : 0);
        }
    }
}