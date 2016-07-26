using System;
using System.Collections.Generic;
using System.Linq;

namespace LearningNeuralNetworks.LearningAlgorithms
{
    public class BackPropagationWithStochasticGradientDescent : LearningAlgorithm
    {
        readonly int epochs;
        readonly int batchSize;
        readonly BackPropagationWithGradientDescent backPropagationWithGradientDescent = new BackPropagationWithGradientDescent();

        public BackPropagationWithStochasticGradientDescent(int epochs, int batchSize)
        {
            this.epochs = epochs;
            this.batchSize = batchSize;
        }

        public override InterpretedNet<TData, TLabel> Apply<TData, TLabel>(InterpretedNet<TData, TLabel> net, IEnumerable<Pair<TData, TLabel>> trainingData, double trainingRateEta, int iterations = 1)
        {
            var rand = new Random();
            var shuffledTrainingData = trainingData.OrderBy(e => rand.Next()).ToArray();
            //
            for (int e = 0; e < epochs; e++)
                for (int batchNo = 0; batchNo * batchSize < shuffledTrainingData.Length; batchNo++)
                {
                    backPropagationWithGradientDescent.Apply(net, shuffledTrainingData.Skip(batchNo * batchSize).Take(batchSize), trainingRateEta);
                }
            return net;
        }
    }
}