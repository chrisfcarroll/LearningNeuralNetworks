using System;
using System.Collections.Generic;
using System.Linq;
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
            for (int i = 0; i < iterations; i++)
            {
                var batch = SelectSample(trainingData, batchSize);
                Apply(net, batch, trainingRateStart / (1+i) );
            }
            return net;
        }

        static V1.Pair<TData,TLabel>[] SelectSample<TData, TLabel>(V1.Pair<TData, TLabel>[] trainingData, int sampleSize)
        {
            var populationSize = trainingData.Length;
            var rnd = new Random();
            var batch = Enumerable
                                    .Range(0, populationSize)
                                    .OrderBy(i => rnd.Next())
                                    .Take(sampleSize)
                                    .Select(i => trainingData[i])
                                    .ToArray();
            return batch;
        }
    }
}