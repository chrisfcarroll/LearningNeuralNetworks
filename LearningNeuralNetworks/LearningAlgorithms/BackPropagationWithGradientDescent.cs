using System.Collections.Generic;
using System.Linq;
using LearningNeuralNetworks.Maths;

namespace LearningNeuralNetworks.LearningAlgorithms
{
    public class BackPropagationWithGradientDescent : LearningAlgorithm
    {
        public override InterpretedNet<TData, TLabel> Apply<TData,TLabel>(InterpretedNet<TData, TLabel> net, IEnumerable<Pair<TData, TLabel>> trainingData, double trainingRateEta)
        {
            foreach (var pair in trainingData)
            {
                var output = net.OutputFor(pair.Data);
                var deltaNablaForNet = ErrorFor(net,pair.Label);
                net.Net.DeltaBiases(deltaNablaForNet.HiddenBiases, deltaNablaForNet.OutputBiases);
                net.Net.DeltaInputToHiddenWeights(deltaNablaForNet.InputToHidden);
                net.Net.DeltaHiddenToOutputWeights(deltaNablaForNet.HiddenToOutput);
            }
            return net;
        }

        public static DeltasFor2Layers ErrorFor<TData, TLabel>(InterpretedNet<TData, TLabel> net, TLabel target)
        {
            return new DeltasFor2Layers();
        }

        public static DeltasFor2Layers ErrorFor(NeuralNet3LayerSigmoid net, ZeroToOne[] target)
        {
            return new DeltasFor2Layers();
        }
    }

    public class DeltasFor2Layers
    {
        public double[] HiddenBiases;
        public double[] OutputBiases;
        public MatrixD InputToHidden;
        public MatrixD HiddenToOutput;

        public override string ToString()
        {
            var hiddenBiases = "[" + HiddenBiases?.Aggregate("", (s, b) => s + ", " + b) + "]";
            var outputBiases = "[" + OutputBiases?.Aggregate("", (s, b) => s + ", " + b) + "]";
            return 
                $"Hidden Biases: {hiddenBiases ?? "none"}\n" + 
                $"OutputBiases: {outputBiases ?? "none"}\n" +
                $"HiddenWeights: {InputToHidden}\n" +
                $"OutputWeights: {HiddenToOutput}\n" ;
        }
    }
}