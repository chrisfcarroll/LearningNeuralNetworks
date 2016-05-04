using System;
using System.Collections.Generic;
using System.Linq;
using MnistParser;

namespace LearningNeuralNetworks.LearningAlgorithms
{
    public class StochasticGradientDescent : LearningAlgorithm
    {
        readonly int epochs;
        readonly int batchSize;
        readonly GradientDescent gradientDescent = new GradientDescent();

        public StochasticGradientDescent(int epochs, int batchSize)
        {
            this.epochs = epochs;
            this.batchSize = batchSize;
        }

        public override InterpretedNet<TData, TLabel> Apply<TData, TLabel>(InterpretedNet<TData, TLabel> net, IEnumerable<Pair<TData, TLabel>> trainingData, double trainingRateEta)
        {
            var rand = new Random();
            var shuffledTrainingData = trainingData.OrderBy(e => rand.Next()).ToArray();
            //
            for (int e = 0; e < epochs; e++)
            for (int batchNo = 0; batchNo * batchSize < shuffledTrainingData.Length; batchNo++)
            {
                gradientDescent.Apply(net, shuffledTrainingData.Skip(batchNo * batchSize).Take(batchSize), trainingRateEta);
            }
            return net;
        }
    }

    public class GradientDescent : LearningAlgorithm
    {
        public override InterpretedNet<TData, TLabel> Apply<TData,TLabel>(InterpretedNet<TData, TLabel> net, IEnumerable<Pair<TData, TLabel>> trainingData, double trainingRateEta)
        {
            var nabla_biases = new double[][] { net.Net.InputLayer.Select(x => x.bias).ToArray(), net.Net.HiddenLayer.Select(x => x.bias).ToArray(), net.Net.OutputLayer.Select(x => x.bias).ToArray() };
            object nabla_weights;
            //
            foreach (var pair in trainingData)
            {
                DeltaNablaForNet deltaNablaForNet = BackPropagate(net,pair);
                net.Net.DeltaBiases(deltaNablaForNet.Biases.InputBiases, deltaNablaForNet.Biases.HiddenBiases, deltaNablaForNet.Biases.OutputBiases);
                net.Net.DeltaInputToHiddenWeights(deltaNablaForNet.Weights.InputToHidden);
                net.Net.DeltaHiddenToOutputWeights(deltaNablaForNet.Weights.HiddenToOutput);
            }
            return net;
        }

        DeltaNablaForNet BackPropagate<TData,TLabel>(InterpretedNet<TData,TLabel> net, Pair<TData,TLabel> pair)
        {
            var result = new DeltaNablaForNet();
            var activation = pair.Data;
            var activations = new List<TData> { pair.Data };

            //forward pass
            net.ActivateInputs(activation);
            //backward pass;
            var labelAsNeuronOutputs = net.ReverseInterpretation(pair.Label);
            var delta = net.Distances(net.Net.LastOutputs, labelAsNeuronOutputs).Select(d=> d * d.SigmoidDerivative());

            return result;
        }

        class DeltaNablaForNet
        {
            public LayerBiases Biases = new LayerBiases();
            public Weights Weights = new Weights();
        }

        class Weights
        {
            public MatrixF InputToHidden;
            public MatrixF HiddenToOutput;
        }
        class LayerBiases
        {
            public double[] InputBiases;
            public double[] HiddenBiases;
            public double[] OutputBiases;
        }
    }
}