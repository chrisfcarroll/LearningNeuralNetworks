using System;
using System.Linq;
using LearningNeuralNetworks.LearningAlgorithms;
using LearningNeuralNetworks.Maths;
using NUnit.Framework;
using TestBase.Shoulds;

namespace LearningNeuralNetworks.Tests.LearningAlgorithms
{
    [TestFixture]
    public partial class TheBackPropagationWithGradientDescentAlgorithm
    {
        [TestFixture]
        public class CalculatesCorrectOutputLayerDeltasFromError
        {
            [TestCase(new[] { 0.35d, 0.9d }, new[] { 0.1d, 0.4d, 0.8d, 0.6d }, new[] { 0.3d, 0.9d }, 0.5d)]
            public void Given__221_SigmoidNetwork(double[] inputs, double[] inputToHiddenWeights, double[] hiddenToOutputWeights, double target)
            {
                var net = NeuralNet3LayerSigmoid.FromFlatWeightArrays(inputs.Length, inputToHiddenWeights, hiddenToOutputWeights);
                var targets = Enumerable.Range(0, net.OutputLayer.Length).Select(i => (ZeroToOne)target);
                var calculatedDeltas = BackPropagationWithGradientDescent.DeltasFor(net, inputs, targets);

                /*
                 * 0.680267196698649d, 0.663738697404353d, 0.690283492907644d
                 */
                var exOutputDelta = -0.040681125112339026d;
                var expected = new DeltasForNeuralNet
                {

                    OutputBiases = new[] { 0d },
                    OutputWeights = new MatrixD(
                        new[,]
                        {
                        {exOutputDelta *  net.HiddenLayer[0].FiringRate },
                        {exOutputDelta *  net.HiddenLayer[1].FiringRate }
                        }),
                };
                calculatedDeltas.OutputWeights.ShouldEqualByValue(expected.OutputWeights);
                calculatedDeltas.OutputBiases.ShouldEqualByValue(expected.OutputBiases);
            }

            [TestCase(new[] { 0.35d, 0.9d }, new[] { 0.1d, 0.4d, 0.8d, 0.6d }, new[] { 0.3d,   0,   0,  /**/  0.9d,   0,    0 }, 0.5d)]
            [TestCase(new[] { 0.35d, 0.9d }, new[] { 0.1d, 0.4d, 0.8d, 0.6d }, new[] { 0.3d,   0, 0.3d, /**/  0.9d,   0,  0.9d }, 0.5d)]
            [TestCase(new[] { 0.35d, 0.9d }, new[] { 0.1d, 0.4d, 0.8d, 0.6d }, new[] { 0.3d, 0.3d,  0,  /**/  0.9d, 0.9d,   0 }, 0.5d)]
            public void Given__223_SigmoidNetwork(double[] inputs, double[] inputToHiddenWeights, double[] hiddenToOutputWeights, double target)
            {
                var net = NeuralNet3LayerSigmoid.FromFlatWeightArrays(inputs.Length, inputToHiddenWeights, hiddenToOutputWeights);
                var targets = Enumerable.Range(0, net.OutputLayer.Length).Select(i => (ZeroToOne)target);
                var calculatedDeltas = BackPropagationWithGradientDescent.DeltasFor(net, inputs, targets);

                var exOutputDelta1 = -0.040681125112339026d;
                var exOutputDelta2 = hiddenToOutputWeights[1].Equals(0) ? 0 : exOutputDelta1;
                var exOutputDelta3 = hiddenToOutputWeights[2].Equals(0) ? 0 : exOutputDelta1;
                var expected = new DeltasForNeuralNet
                {

                    OutputBiases = new[] { 0d, 0d, 0d },
                    OutputWeights = new MatrixD(
                        new[,]
                        {
                            {exOutputDelta1 *  net.HiddenLayer[0].FiringRate, exOutputDelta2 * net.HiddenLayer[0].FiringRate, exOutputDelta3 * net.HiddenLayer[0].FiringRate },
                            {exOutputDelta1 *  net.HiddenLayer[1].FiringRate, exOutputDelta2 * net.HiddenLayer[1].FiringRate, exOutputDelta3 * net.HiddenLayer[1].FiringRate }
                        }),
                };
                calculatedDeltas.OutputWeights.ShouldEqualByValue(expected.OutputWeights);
                calculatedDeltas.OutputBiases.ShouldEqualByValue(expected.OutputBiases);
            }

        }

        int CountHits(InterpretedNet<string, int> guineaPig, Pair<string, int>[] trainingData)
        {
            return trainingData.Count(p => guineaPig.OutputFor(p.Data) == p.Label);
        }

        Pair<string, int>[] GenerateTrainingData()
        {
            var rnd = new Random();
            var vowels = new[] {'A', 'E', 'I', 'O', 'U'};
            var trainingData = new Pair<string, int>[1000];
            for (int i = 0; i < 1000; i++)
            {
                var data = new string(Enumerable.Range(0, 10).Select(x => (char) rnd.Next('A', 'Z')).ToArray());
                var label = data.Count(c => vowels.Contains(c));
                trainingData[i] = new Pair<string, int>(data, label);
            }
            return trainingData;
        }
    }
}