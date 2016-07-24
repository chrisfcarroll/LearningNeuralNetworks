using System.Collections.Generic;
using System.Linq;

namespace LearningNeuralNetworks
{
    public static class InputNeuronBuilder
    {
        public static ActivationFunction Id = x => x;
        public static Neuron FixedSensorOn() { return new Neuron { bias = 1, ActivationFunction = Id}; }
        public static Neuron FixedSensorOff() { return new Neuron { bias = 0, ActivationFunction = Id}; }
        public static Neuron FixedSensor(double fixedBias) { return new Neuron { bias = fixedBias, ActivationFunction = Id}; }
    }

    public static class SigmoidNeuronBuilder
    {

        public static Neuron Nand(params Neuron[] inputs) { return Nand((IEnumerable<Neuron>) inputs); }

        public static Neuron Nand(IEnumerable<Neuron> inputs)
        {
            var num = inputs.Count();
            var nand = new Neuron
            {
                bias = num * Neuron.High,
                Inputs = inputs.Select(p => new Sinput(p, -Neuron.High)).ToArray(),
                ActivationFunction = SigmoidNeuronExtensionMethods.Sigmoid
            };
            return nand;
        }

        public static Neuron Or(params Neuron[] inputs) { return Or((IEnumerable<Neuron>)inputs); }

        public static Neuron Or(IEnumerable<Neuron> inputs)
        {
            return new Neuron
            {
                bias = 0,
                Inputs = inputs.Select(p => new Sinput(p, Neuron.High)).ToArray(),
                ActivationFunction = SigmoidNeuronExtensionMethods.Sigmoid
            };
        }
    }
}