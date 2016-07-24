using System.Collections.Generic;
using System.Linq;
using LearningNeuralNetworks.Maths;

namespace LearningNeuralNetworks
{
    public static class SigmoidNeuronBuilder
    {

        public static Neuron Nand(params Neuron[] inputs) { return Nand((IEnumerable<Neuron>) inputs); }

        public static Neuron Nand(IEnumerable<Neuron> inputs)
        {
            var num = inputs.Count();
            var nand = new Neuron(MathExt.Sigmoid)
            {
                Bias = num * Neuron.High,
                Inputs = inputs.Select(p => new Ninput(p, -Neuron.High)).ToArray()
            };
            return nand;
        }

        public static Neuron Or(params Neuron[] inputs) { return Or((IEnumerable<Neuron>)inputs); }

        public static Neuron Or(IEnumerable<Neuron> inputs)
        {
            return new Neuron(MathExt.Sigmoid)
            {
                Bias = 0,
                Inputs = inputs.Select(p => new Ninput(p, Neuron.High)).ToArray()
            };
        }
    }
}