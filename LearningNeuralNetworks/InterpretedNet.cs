using System;
using System.Collections.Generic;
using MnistParser;

namespace LearningNeuralNetworks
{
    public class InterpretedNet<T>
    {
        public Func<IEnumerable<ZeroToOne>, T> Interpretation { get; set; }
        public Func<T, IEnumerable<ZeroToOne>> ReverseInterpretation { get; set; }
        public IComparer<T> Comparer { get; set; }
        public NeuralNet3LayerSigmoid Net { get; }

        public NeuralNet3LayerSigmoid LearnFrom(IEnumerable<MNistPair> trainingData, double trainingRateEta, LearningAlgorithm<T> algorithm)
        {
            algorithm.Apply(this, trainingData, trainingRateEta);
            return this;
        }

        public T LastOutput => Net.LastOutputAs(Interpretation);

        public T OutputFor(double[] input)
        {
            Net.ActivateInputs(input);
            return LastOutput;
        }

        public InterpretedNet(NeuralNet3LayerSigmoid net, Func<IEnumerable<ZeroToOne>, T> interpretation, Func<T, IEnumerable<ZeroToOne>> reverseInterpretation, IComparer<T> comparer)
        {
            Interpretation = interpretation;
            ReverseInterpretation = reverseInterpretation;
            Comparer = comparer;
            this.Net = net;
        }

        public static implicit operator NeuralNet3LayerSigmoid(InterpretedNet<T> interpretedNet) { return interpretedNet.Net;}
    }
}