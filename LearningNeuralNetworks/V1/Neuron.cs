using System.Linq;
using LearningNeuralNetworks.Maths;

namespace LearningNeuralNetworks.V1
{
    public delegate double ActivationFunction(double inputValue);

    /// <summary>
    /// Models a Sigmoid Neuron, see e.g. <see href="http://neuralnetworksanddeeplearning.com/chap1.html#sigmoid_neurons">this introduction</see>.
    /// </summary>
    public partial class Neuron
    {
        public const double High = 1e100d;
        public const double Low = -High;

        public double Bias;
        public Ninput[] Inputs;
        public ActivationFunction ActivationFunction;

        public double FiringRate => ActivationFunction(Input);
        internal double Input => Inputs.DotProduct() + Bias;

        public string ToString(string format)
        {
            var inputs = Inputs != null
                             ? string.Join(",", Inputs.Select(i => i.Weight.ToString(format))) + "=>"
                             : "";
            return
                inputs +
                (ActivationFunction == Id
                     ? "Id=>"
                     : ActivationFunction == MathExt.Sigmoid
                           ? "Sigmoid=>"
                           : "") +
                FiringRate.ToString(format);
        }

        public override string ToString()
        {
            return ToString(""); 
        }
    }


    ///<summary>Models a weighted input to a <see cref="Neuron"/> </summary>
    public struct Ninput
    {
        public double Weight;
        public Neuron Source;
        public Ninput(Neuron source, double weight) : this()
        {
            Weight = weight;
            Source = source;
        }
    }
}