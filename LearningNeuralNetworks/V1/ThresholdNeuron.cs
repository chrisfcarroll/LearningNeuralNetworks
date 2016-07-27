using System.Collections.Generic;

namespace LearningNeuralNetworks.V1
{
    /// <summary>
    /// Models a Perceptron, see e.g. <see href="http://neuralnetworksanddeeplearning.com/chap1.html#perceptrons">this introduction</see>. 
    /// With integers, not Reals
    /// </summary>
    public struct ThresholdNeuron
    {
        public int bias;
        public IList<Input> Inputs;

        public bool IsFiring => Inputs.DotProduct() + bias > 0;

        public struct Input
        {
            public int weight;
            public ThresholdNeuron Source;
            public Input(ThresholdNeuron source, int weight) : this()
            {
                this.weight = weight;
                this.Source = source;
            }
        }
    }

}