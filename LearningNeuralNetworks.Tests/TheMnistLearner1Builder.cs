using System;
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
            net.Net.InputLayer.Length.ShouldBe(Image.ByteSize);
            net.Net.HiddenLayer.Length.ShouldBe(15);
            net.Net.OutputLayer.Length.ShouldBe(10);
        }

        [Test]
        public void BuildsANetworkWithInterpretationOfOutputLayerAsADigit()
        {
            var net = MnistSigmoidLearner1NetBuilderTrainer.Build(15);
            net.OutputFor(new[] {1d}).ShouldBeOfType<byte>();
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
            var net = MnistSigmoidLearner1NetBuilderTrainer.Build(15);
            //
            var digitsAsOutputNeurons =
                Enumerable.Range(0, 10)
                    .Select(i => Enumerable.Range(0, 10).Select(j => j == i ? new ZeroToOne(1) : new ZeroToOne(0))).ToArray();

            for (byte i = 0; i < 10; i++)
            {
                net.ReverseInterpretation(i).ShouldEqualByValue(digitsAsOutputNeurons[i]);
            }
        }


        [TestFixture]
        public class IsTrainableAnd
        {

            [Test]
            public void CanLearnSomehow()
            {
                var net = MnistSigmoidLearner1NetBuilderTrainer.Build(15);
                var trainingData = new MnistFilesReader(mnistRealDataDirectory).TrainingData.Take(100);
                var scoreBeforeTraining = HitsScoredOnTestData(net, trainingData);
                //
                net.LearnFrom(trainingData, 0.01, new RandomFall<byte>());
                //
                var scoreAfterTraining = HitsScoredOnTestData(net, trainingData.Take(100));
                scoreAfterTraining.ShouldBeGreaterThan(scoreBeforeTraining);
            }

            int HitsScoredOnTestData(NeuralNet3LayerSigmoid net, IEnumerable<MNistPair> testData)
            {
                return
                    testData.Count(
                        d =>
                            net.ActivateInputs(d.Image.As1Ddoubles)   
                               .LastOutputAs(o => o.ArgMaxIndex(e => e)) == d.Label);
            }
        }
    }

}
