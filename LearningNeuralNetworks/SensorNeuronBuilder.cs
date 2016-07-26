using System.Collections.Generic;
using System.Linq;

namespace LearningNeuralNetworks
{
    /// <summary>
    /// Builds Sensor Neurons - the activation function is the identity function. Used for the input layer of a network.
    /// </summary>
    public static class SensorNeuronBuilder
    {
        /// <returns>A Sensor neuron with Bias=1</returns>
        public static Neuron On() { return new Neuron(Neuron.Id){ Bias = 1}; }

        /// <returns>A Sensor neuron with Bias=0</returns>
        public static Neuron Off() { return new Neuron(Neuron.Id) { Bias = 0}; }

        /// <returns>A Sensor neuron with Bias= <paramref name="fixedBias"/></returns>
        public static Neuron WithValue(ZeroToOne fixedBias) { return new Neuron(Neuron.Id){ Bias = fixedBias}; }

        /// <returns>An IEnumerable of Sensor neurons, with biases preset to <paramref name="biases"/></returns>
        public static IEnumerable<Neuron> WithValues(IEnumerable<double> biases) { return biases.Select(b=> new Neuron(Neuron.Id){Bias = b}); }
    }
}