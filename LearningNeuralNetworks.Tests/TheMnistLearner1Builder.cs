﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MnistParser;
using NUnit.Framework;
using TestBase.Shoulds;

namespace LearningNeuralNetworks.Tests
{
    [TestFixture]
    public class TheMnistSigmoidLearner1NetBuilderTrainer
    {
        static string mnistRealDataDirectory = @"..\..\..\MnistParser\MnistData";

        [Test]
        public void BuildsANetworkWith784InputNeurons10OutputNeuronsAndTheRequestedHiddenLayerSize()
        {
            var net= MnistSigmoidLearner1NetBuilderTrainer.Build(15);
            net.InputLayer.Length.ShouldBe(MnistParser.Image.ByteSize);
            net.HiddenLayer.Length.ShouldBe(15);
            net.OutputLayer.Length.ShouldBe(10);
        }

        [Test]
        public void TrainsItsNetWithStochasticGradientDescent()
        {
            var net = MnistSigmoidLearner1NetBuilderTrainer.Build(15);
            var trainingData = new MnistFilesReader(mnistRealDataDirectory);
            var scoreBeforeTraining = HitsScoredOnTestData(net, trainingData.TrainingData.Take(100));
            //
            net.LearnWithGradientDescentStochasticallyFrom(trainingData.TrainingData, 10, 10, 0.1d);
            //
            var scoreAfterTraining = HitsScoredOnTestData(net, trainingData.TrainingData.Take(100));
            scoreAfterTraining.ShouldBeGreaterThan(scoreBeforeTraining);
        }

        public int HitsScoredOnTestData(NeuralNet3LayerSigmoid net, IEnumerable<MNistPair> testData)
        {
            return testData.Count(d => net.OutputFor(d.Image.As1D.Select(b=>(double)b).ToArray()).ArgMaxIndex(e=>e) == d.Label);
        }
    }

}
