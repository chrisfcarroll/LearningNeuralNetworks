using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MnistParser;

namespace LearningNeuralNetworks.LearningAlgorithms
{
    /// <summary>
    /// An erratic "learning" algorithm, a bit like random walk, but trying to fall downhill.
    /// </summary>
    /// <typeparam name="TLabel"></typeparam>
    public class RandomWalkFall : LearningAlgorithm
    {
        public int Iterations { get;}

        public override InterpretedNet<TData, TLabel> Apply<TData, TLabel>(InterpretedNet<TData, TLabel> net, IEnumerable<Pair<TData, TLabel>> trainingData, double trainingRateEta)
        {
            foreach (var pair in trainingData)
            {
                MutateNetByRandomFalls(net, trainingRateEta, pair);
            }
            return net;
        }

        int MutateNetByRandomFalls<TData, TLabel>(InterpretedNet<TData, TLabel> net, double trainingRateEta, Pair<TData, TLabel> pair)
        {
            var falls = 0;
            var bestSoFar = net.Net.OutputFor(net.InputEncoding(pair.Data));
            if (!net.OutputInterpretation(bestSoFar).Equals(pair.Label))
            {
                for (int e = 0; e < Iterations; e++)
                {
                    var deltaInputToHidden = net.Net.InputToHidden.Copy();
                    for (int i = 0; i < deltaInputToHidden.RowCount; i++)
                        for (int j = 0; j < deltaInputToHidden.ColumnCount; j++)
                        {
                            if (rnd.NextDouble() < trainingRateEta)
                                deltaInputToHidden[i, j] = Randomize(deltaInputToHidden[i, j]);
                        }
                    var deltaHiddenToOutput = net.Net.HiddenToOutput.Copy();
                    for (int i = 0; i < deltaHiddenToOutput.RowCount; i++)
                        for (int j = 0; j < deltaHiddenToOutput.ColumnCount; j++)
                        {
                            if (rnd.NextDouble() < trainingRateEta)
                                deltaHiddenToOutput[i, j] = Randomize(deltaHiddenToOutput[i, j]);
                        }
                    var deltaHiddenBiases = net.Net.HiddenLayer.Select(n => Randomize(n.Bias));
                    var deltaOutputBiases = net.Net.OutputLayer.Select(n => Randomize(n.Bias));
                    //
                    net.Net.DeltaInputToHiddenWeights(deltaInputToHidden);
                    net.Net.DeltaHiddenToOutputWeights(deltaHiddenToOutput);
                    net.Net.DeltaBiases(deltaHiddenBiases, deltaOutputBiases);

                    var newResult = net.Net.OutputFor(net.InputEncoding(pair.Data));

                    if (net.CloserOutputToTarget(pair.Label, newResult, bestSoFar).Equals(bestSoFar))
                    {
                        //then revert
                        net.Net.DeltaInputToHiddenWeights(-deltaInputToHidden);
                        net.Net.DeltaHiddenToOutputWeights(deltaHiddenToOutput);
                        net.Net.DeltaBiases(deltaHiddenBiases, deltaOutputBiases);
                    }
                    else
                    {
                        falls++;
                    }
                }
            }
            return falls;
        }

        readonly Random rnd = new Random();
        double Randomize(double input) { return  input==0 ? rnd.Next(100)-50 : input * (rnd.Next(10) - 5) * rnd.NextDouble(); }

        public RandomWalkFall(int iterations) { Iterations = iterations; }
        public RandomWalkFall() { Iterations = 3; }
    }
}