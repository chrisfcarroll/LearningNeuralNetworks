using System;
using System.Collections.Generic;
using System.Linq;
using LearningNeuralNetworks.LearningAlgorithms;
using LearningNeuralNetworks.V1;
using MnistParser;
using NUnit.Framework;
using TestBase.Shoulds;

namespace LearningNeuralNetworks.Tests.V1
{
    [TestFixture]
    public class TheMnistSigmoidLearner1NetBuilderTrainer
    {
        static string mnistRealDataDirectory = @"..\..\..\MnistParser\MnistData";

        [Test]
        public void BuildsANetworkWith784InputNeurons10OutputNeuronsAndTheRequestedHiddenLayerSize()
        {
            var net= MnistLearnerSigmoidNetBuilder.Build(15);
            net.Net.InputLayer.Length.ShouldBe(Image.ByteSize);
            net.Net.HiddenLayer.Length.ShouldBe(15);
            net.Net.OutputLayer.Length.ShouldBe(10);
        }

        [Test]
        public void BuildsANetworkWithInterpretationOfOutputLayerAsADigit()
        {
            var net = MnistLearnerSigmoidNetBuilder.Build(15);
            net.OutputFor(new[] { 1d}).ShouldBeOfType<byte>();
            //
            var digitsAsOutputNeurons =
                Enumerable.Range(0, 10)
                    .Select(i => Enumerable.Range(0, 10).Select(j => j == i ? new ZeroToOne(1) : new ZeroToOne(0))).ToArray();

            digitsAsOutputNeurons[0].ShouldEqualByValue(new ZeroToOne[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
            digitsAsOutputNeurons[1].ShouldEqualByValue(new ZeroToOne[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 });
            digitsAsOutputNeurons[4].ShouldEqualByValue(new ZeroToOne[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 });
            digitsAsOutputNeurons[5].ShouldEqualByValue(new ZeroToOne[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 });
            digitsAsOutputNeurons[9].ShouldEqualByValue(new ZeroToOne[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 });
        }

        [Test]
        public void BuildsANetworkWithReverseInterpretationBeingTheInverseOfTheInterpretation()
        {
            var net = MnistLearnerSigmoidNetBuilder.Build(15);
            //
            var digitsAsOutputNeurons =
                Enumerable.Range(0, 10)
                    .Select(i => Enumerable.Range(0, 10).Select(j => j == i ? new ZeroToOne(1) : new ZeroToOne(0))).ToArray();

            for (byte i = 0; i < 10; i++)
            {
                net.ReverseInterpretation(i).ShouldEqualByValue(digitsAsOutputNeurons[i]);
            }
        }


        [Test,Ignore("BP not yet working")]
        public void IsTrainableByBackPropagation()
        {
            var net = MnistLearnerSigmoidNetBuilder.Build(15);
            var trainingData = new MnistFilesReader(mnistRealDataDirectory).TrainingData.Take(200).Select(p => new LearningNeuralNetworks.V1.Pair<Image, byte>(p.Data, p.Label)).ToArray();
            var scoreBeforeTraining = HitsScoredOnTestData(net, trainingData);
            //
            new BackPropagationWithGradientDescent().ApplyToBatches(net, trainingData, 10, 3, 100);
            //
            var scoreAfterTraining = HitsScoredOnTestData(net, trainingData);
            Console.WriteLine("Scores before/after training: {0} / {1}", scoreBeforeTraining, scoreAfterTraining);
            scoreAfterTraining.ShouldBeGreaterThan(scoreBeforeTraining);
        }

        int HitsScoredOnTestData(InterpretedNet<Image, byte> net, IEnumerable<LearningNeuralNetworks.V1.Pair<Image,byte>> testData)
        {
            return testData.Count(d => net.OutputFor(d.Data) == d.Label);
        }
    }

}
