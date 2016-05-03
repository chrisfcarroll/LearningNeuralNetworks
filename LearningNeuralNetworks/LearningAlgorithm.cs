using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MnistParser;

namespace LearningNeuralNetworks
{
    public class LearningAlgorithm
    {
        public virtual LearningAlgorithm Apply<TData,TLabel>(InterpretedNet<TData,TLabel> net, IEnumerable<Pair<TData,TLabel>> trainingData, double trainingRateEta)
        {
            return this;
        }
    }

    /// <summary>
    /// A bit like random walk, but trying to walk downhill.
    /// </summary>
    /// <typeparam name="TLabel"></typeparam>
    public class RandomWalkFall : LearningAlgorithm
    {
        public int Iterations { get;}

        public override LearningAlgorithm Apply<TData, TLabel>(InterpretedNet<TData,TLabel> net, IEnumerable<Pair<TData,TLabel>> trainingData, double trainingRateEta)
        {
            Debug.Assert( typeof(TData) == typeof(Image), "Sorry, only coded Images at the moment");

            foreach (var pair in trainingData)
            {
                var bestSoFar = net.Net.OutputFor( net.InputEncoding(pair.Data));
                if ( !net.OutputInterpretation(bestSoFar).Equals(pair.Label))
                {
                    for (int e = 0; e < Iterations; e++)
                    {
                        var deltaInputToHidden = net.Net.InputToHidden.Copy();
                        for (int i = 0; i < deltaInputToHidden.RowCount; i++)
                        for (int j = 0; j < deltaInputToHidden.ColumnCount; j++)
                        {
                            if (rnd.NextDouble() < trainingRateEta) deltaInputToHidden[i, j] = Randomize(deltaInputToHidden[i, j]);
                        }
                        var deltaHiddenToOutput = net.Net.HiddenToOutput.Copy();
                        for (int i = 0; i < deltaHiddenToOutput.RowCount; i++)
                        for (int j = 0; j < deltaHiddenToOutput.ColumnCount; j++)
                        {
                            if (rnd.NextDouble() < trainingRateEta) deltaHiddenToOutput[i, j] = Randomize(deltaHiddenToOutput[i, j]);
                        }
                        var deltaInputBiases = net.Net.InputLayer.Select(n => Randomize(n.bias));
                        var deltaHiddenBiases = net.Net.HiddenLayer.Select(n => Randomize(n.bias));
                        var deltaOutputBiases = net.Net.OutputLayer.Select(n => Randomize(n.bias));
                        //
                        net.Net.DeltaInputToHiddenWeights(deltaInputToHidden);
                        net.Net.DeltaHiddenToOutputWeights(deltaHiddenToOutput);
                        net.Net.DeltaBiases(deltaInputBiases, deltaHiddenBiases, deltaOutputBiases);

                        var newResult = net.Net.OutputFor(net.InputEncoding(pair.Data));

                        if ( net.Closest(pair.Label, newResult, bestSoFar).Equals(bestSoFar) )
                        {
                            //then revert
                            net.Net.DeltaInputToHiddenWeights(-deltaInputToHidden);
                            net.Net.DeltaHiddenToOutputWeights(deltaHiddenToOutput);
                            net.Net.DeltaBiases(deltaInputBiases, deltaHiddenBiases, deltaOutputBiases);
                        }
                    }
                }
            }
            return this;
        }

        readonly Random rnd = new Random();
        double Randomize(double input) { return  input==0 ? rnd.Next(100)-50 : input * (rnd.Next(10) - 5) * rnd.NextDouble(); }

        public RandomWalkFall(int iterations) { Iterations = iterations; }
        public RandomWalkFall() { Iterations = 2; }
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

        public override LearningAlgorithm Apply<TData, TLabel>(InterpretedNet<TData,TLabel> net, IEnumerable<Pair<TData,TLabel>> trainingData, double trainingRateEta)
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
        public override LearningAlgorithm Apply<TData,TLabel>(InterpretedNet<TData,TLabel> net, IEnumerable<Pair<TData,TLabel>> trainingData, double trainingRateEta)
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