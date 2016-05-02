using System;
using System.Collections.Generic;
using System.Linq;
using MnistParser;

namespace LearningNeuralNetworks
{
    public class LearningAlgorithm
    {
        public virtual LearningAlgorithm Apply(NeuralNet3LayerSigmoid net, IEnumerable<MNistPair> trainingData, double trainingRateEta)
        {
            return this;
        }
    }

    public class StochasticGradientDescent : LearningAlgorithm
    {
        readonly int epochs;
        readonly int batchSize;
        readonly GradientDescent gradientDescent= new GradientDescent();

        public StochasticGradientDescent(int epochs, int batchSize)
        {
            this.epochs = epochs;
            this.batchSize = batchSize;
        }

        public override LearningAlgorithm Apply(NeuralNet3LayerSigmoid net, IEnumerable<MNistPair> trainingData, double trainingRateEta)
        {
            var rand = new Random();
            var shuffledTrainingData = trainingData.OrderBy(e => rand.Next()).ToArray();
            //
            for (int e = 0; e < epochs; e++)
                for (int batchNo = 0; batchNo*batchSize < shuffledTrainingData.Length; batchNo++)
                {
                    gradientDescent.Apply(net, shuffledTrainingData.Skip(batchNo*batchSize).Take(batchSize), trainingRateEta);
                }
            return this;
        }
    }
    public class GradientDescent : LearningAlgorithm
    {
        public override LearningAlgorithm Apply(NeuralNet3LayerSigmoid net, IEnumerable<MNistPair> trainingData, double trainingRateEta)
        {
            var nabla_biases = new double[][] { net.InputLayer.Select(x => x.bias).ToArray(), net.HiddenLayer.Select(x => x.bias).ToArray(), net.OutputLayer.Select(x => x.bias).ToArray() };
            object nabla_weights;
            //
            foreach (var pair in trainingData)
            {
                DeltaNablaForNet deltaNablaForNet = BackPropagate(net,pair);
                net.DeltaBiases(deltaNablaForNet.Biases.InputBiases, deltaNablaForNet.Biases.HiddenBiases, deltaNablaForNet.Biases.OutputBiases);
                net.DeltaInputToHiddenWeights(deltaNablaForNet.Weights.InputToHidden);
                net.DeltaHiddenToOutputWeights(deltaNablaForNet.Weights.HiddenToOutput);
            }
            return this;
        }

        DeltaNablaForNet BackPropagate(NeuralNet3LayerSigmoid net, MNistPair pair)
        {
            var result = new DeltaNablaForNet();
            var activation = pair.Image.As1D;
            var activations = new List<Image> { pair.Image };

            //forward pass
            net.ActivateInputs(activation);
            //backward pass;
            var delta = net.LastOutputs.Select(x => (x - pair.Label) * x.SigmoidDerivative());

            return result;
        }

        class DeltaNablaForNet
        {
            public LayerBiases Biases = new LayerBiases();
            public Weights Weights = new Weights();
        }

        class Weights
        {
            public double[,] InputToHidden;
            public double[,] HiddenToOutput;
        }
        class LayerBiases
        {
            public double[] InputBiases;
            public double[] HiddenBiases;
            public double[] OutputBiases;
        }
    }


}