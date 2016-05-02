using System;
using System.Collections.Generic;
using System.Linq;
using MnistParser;

namespace LearningNeuralNetworks
{
    public class LearningAlgorithm<T>
    {
        public virtual LearningAlgorithm<T> Apply(InterpretedNet<T> net, IEnumerable<MNistPair> trainingData, double trainingRateEta)
        {
            return this;
        }
    }

    /// <summary>
    /// A bit like random walk, but trying to walk downhill.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class RandomFall<T> : LearningAlgorithm<T>
    {
        public int Iterations { get;}

        public override LearningAlgorithm<T> Apply(InterpretedNet<T> net, IEnumerable<MNistPair> trainingData, double trainingRateEta)
        {
            foreach (var pair in trainingData)
            {
                var bestSoFar = net.OutputFor(pair.Image.As1Ddoubles);
                if (!bestSoFar.Equals(pair.Label))
                {
                    var rnd = new Random();
                    Func<double, double> randomize = input => input*(rnd.Next(10) - 5)*rnd.NextDouble();
                    for (int e = 0; e < Iterations; e++)
                    {
                        var deltaInputToHidden = net.Net.InputToHidden.Clone();
                        for (int i = 0; i < deltaInputToHidden.RowCount; i++)
                            for (int j = 0; j < deltaInputToHidden.ColumnCount; j++)
                            {
                                deltaInputToHidden[i, j] = randomize(deltaInputToHidden[i, j]);
                            }
                        var deltaHiddenToOutput = net.Net.HiddenToOutput.Clone();
                        for (int i = 0; i < deltaHiddenToOutput.RowCount; i++)
                            for (int j = 0; j < deltaHiddenToOutput.ColumnCount; j++)
                            {
                                deltaHiddenToOutput[i, j] = randomize(deltaHiddenToOutput[i, j]);
                            }
                        var deltaInputBiases = net.Net.InputLayer.Select(n => randomize(n.bias));
                        var deltaHiddenBiases = net.Net.HiddenLayer.Select(n => randomize(n.bias));
                        var deltaOutputBiases = net.Net.OutputLayer.Select(n => randomize(n.bias));
                        //
                        net.Net.DeltaInputToHiddenWeights(deltaInputToHidden);
                        net.Net.DeltaHiddenToOutputWeights(deltaHiddenToOutput);
                        net.Net.DeltaBiases(deltaInputBiases, deltaHiddenBiases, deltaOutputBiases);

                        var newResult = net.OutputFor(pair.Image.As1Ddoubles);
                    }
                }
            }
            return this;
        }

        public RandomFall(int iterations) { Iterations = iterations; }
        public RandomFall() { Iterations = 2; }
    }

    public class StochasticGradientDescent<T> : LearningAlgorithm<T>
    {
        readonly int epochs;
        readonly int batchSize;
        readonly GradientDescent<T> gradientDescent= new GradientDescent<T>();

        public StochasticGradientDescent(int epochs, int batchSize)
        {
            this.epochs = epochs;
            this.batchSize = batchSize;
        }

        public override LearningAlgorithm<T> Apply(InterpretedNet<T> net, IEnumerable<MNistPair> trainingData, double trainingRateEta)
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
    public class GradientDescent<T> : LearningAlgorithm<T>
    {
        public override LearningAlgorithm<T> Apply(InterpretedNet<T> net, IEnumerable<MNistPair> trainingData, double trainingRateEta)
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
            return this;
        }

        DeltaNablaForNet BackPropagate(InterpretedNet<T> net, MNistPair pair)
        {
            var result = new DeltaNablaForNet();
            var activation = pair.Image.As1DBytes;
            var activations = new List<Image> { pair.Image };

            //forward pass
            net.Net.ActivateInputs(activation);
            //backward pass;
            var delta = net.Net.LastOutputs.Select(x => (x - pair.Label) * x.SigmoidDerivative());

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