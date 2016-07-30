using System;
using System.Collections.Generic;
using System.Linq;
using LearningNeuralNetworks.Frameworks;
using MnistParser;

namespace LearningNeuralNetworks.LearningAlgorithms
{
    /// <summary>
    /// A learning algorithm is expected to mutate the given neural net, presumably by training it with the given data.
    /// </summary>
    public class LearningAlgorithm
    {
        /// <param name="net">The neural net, wrapped with interpretations of Data to Input weights, output weights to Label, and Label to output weights</param>
        /// <param name="trainingData">The labelled training data</param>
        /// <param name="trainingRateEta">expected to be between 0 and 1 for most algorithms, but algoritms may interpret it as they see fit.</param>
        /// <param name="iterations"></param>
        public virtual InterpretedNet<TData, TLabel> Apply<TData,TLabel>(InterpretedNet<TData, TLabel> net, IEnumerable<V1.Pair<TData, TLabel>> trainingData, double trainingRateEta, int iterations = 1)
        {
            return net;
        }

        public virtual InterpretedNet<TData, TLabel> ApplyToBatches<TData, TLabel>(InterpretedNet<TData, TLabel> net, V1.Pair<TData, TLabel>[] trainingData, int batchSize, double trainingRateStart, int iterations = 1)
        {
            var trainingDataInRandomOrder = trainingData.OrderRandomly();

            for (int i = 0; i < iterations; i++)
            {
                var batch = trainingDataInRandomOrder.Skip(i*batchSize).Take(batchSize);
                Apply(net, batch, trainingRateStart / (1+i) );
            }
            return net;
        }

    }
}