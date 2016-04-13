using System.Collections.Generic;

namespace LearningNeuralNetworks
{
    /// <summary>
    /// Models a Perceptron, see e.g. <see href="http://neuralnetworksanddeeplearning.com/chap1.html#perceptrons">this introduction</see>. 
    /// With integers, not Reals
    /// </summary>
    public struct Perceptron
    {
        public int bias;
        public IList<Pinput> Inputs;

        public bool IsFiring => Inputs.DotProduct() + bias > 0;
    }

    public struct Pinput
    {
        public int weight;
        public Perceptron Source;
        public Pinput(Perceptron source, int weight) : this()
        {
            this.weight = weight;
            this.Source = source;
        }
    }
}