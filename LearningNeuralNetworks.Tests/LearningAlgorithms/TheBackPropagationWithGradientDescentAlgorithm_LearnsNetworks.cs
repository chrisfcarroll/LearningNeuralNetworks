﻿using System;
using System.Linq;
using LearningNeuralNetworks.LearningAlgorithms;
using NUnit.Framework;
using TestBase.Shoulds;

namespace LearningNeuralNetworks.Tests.LearningAlgorithms
{
    [TestFixture]
    public partial class TheBackPropagationWithGradientDescentAlgorithm
    {
        [TestFixture]
        public class LearnsNetworks
        {
            [TestCase(10,  5, 1, 7000)]
            [TestCase(10, 10, 1, 4000)]
            [TestCase(10, 20, 1, 3000)]
            public void Given__ASigmoidNetworkOfSuitableSize__AndSomeTrainingData(int inputLayerSize, int hiddenLayerSize, int outputLayerSize, int trainingSamplesCount)
            {
                var rawNet = new NeuralNet3LayerSigmoid(inputLayerSize, hiddenLayerSize, outputLayerSize);
                var interpetedNet= new InterpretedNet<string,int>(
                    rawNet,
                    s => s.Select(c=> (ZeroToOne)(1.0d * c/256) ).ToArray(),
                    o => (int)(o.Single() * 100),
                    i => new [] { (ZeroToOne)(i / 100d)},
                    (x,y) => x.Zip(y, (xx,yy)=> Math.Abs(xx-yy))
                    );
                var trainingData = GenerateRandomDataAndLabels(trainingSamplesCount);
                var testData = GenerateRandomDataAndLabels(10 + trainingSamplesCount/10);
                var scoreBeforeTraining = CountHits(interpetedNet, testData);

                //
                new BackPropagationWithGradientDescent().Apply(interpetedNet, trainingData, 0.1);
                var scoreAfterTraining = CountHits(interpetedNet, testData);

                //
                Console.Write("Hits for NN size {3}, {4}, {5} Before / After training with {0} samples : {1} / {2}",
                        trainingSamplesCount, 
                        scoreBeforeTraining, scoreAfterTraining,
                        inputLayerSize, hiddenLayerSize, outputLayerSize);

                scoreAfterTraining.Data.ShouldBeGreaterThan(scoreBeforeTraining.Data * 1.1);
            }

            Pair<int,int> CountHits(InterpretedNet<string, int> guineaPig, Pair<string, int>[] testData)
            {
                return new Pair<int, int>(
                    testData.Count(p => guineaPig.OutputFor(p.Data) == p.Label),
                    testData.Length);
            }

            Pair<string, int>[] GenerateRandomDataAndLabels(int howManySamples)
            {
                var rnd = new Random();
                var vowels = new[] { 'A', 'E', 'I', 'O', 'U' };
                var trainingData = new Pair<string, int>[howManySamples];
                for (int i = 0; i < howManySamples; i++)
                {
                    var data = new string(Enumerable.Range(0, 10).Select(x => (char)rnd.Next('A', 'Z')).ToArray());
                    var label = data.Count(c => vowels.Contains(c));
                    trainingData[i] = new Pair<string, int>(data, label);
                }
                return trainingData;
            }
        }
    }
}