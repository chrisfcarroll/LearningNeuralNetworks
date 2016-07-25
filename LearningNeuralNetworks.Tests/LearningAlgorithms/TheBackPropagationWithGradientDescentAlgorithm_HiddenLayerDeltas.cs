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
        public class CalculatesCorrectHiddenLayerDeltasFromError
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
                var outputs = net.LastOutputs.ToArray();
                var exOutputDelta = -0.040681125112339026d;
                var exHiddenDelta0 = -0.0023685025015371172d;
                var exHiddenDelta1 = -0.0075927347073177845d;
                var expectedOutputDeltas = new DeltasForNeuralNet
                {

                    OutputBiases = new[] { 0d },
                    OutputWeights = new MatrixD(
                        new[,]
                        {
                        {exOutputDelta *  net.HiddenLayer[0].FiringRate },
                        {exOutputDelta *  net.HiddenLayer[1].FiringRate }
                        }),
                    HiddenBiases = new [] {0d, 0d},
                    HiddenWeights = new MatrixD(
                        new[,]
                        {
                            {
                                exHiddenDelta0 *  net.HiddenLayer[0].Inputs[0].Source.FiringRate,
                                exHiddenDelta1 * net.HiddenLayer[1].Inputs[0].Source.FiringRate 
                            },
                            {
                                exHiddenDelta0 *  net.HiddenLayer[0].Inputs[1].Source.FiringRate,
                                exHiddenDelta1 * net.HiddenLayer[1].Inputs[1].Source.FiringRate 
                            }
                        }),
                };
                var expectedDeltaValues = new DeltasForNeuralNet
                {

                    OutputBiases = new[] { 0d },
                    OutputWeights = new MatrixD(
                        new[,]
                        {
                            {-0.040681125112339026d *  0.680267196698649d  }, // -0.0276740349387179
                            {-0.040681125112339026d *  0.663738697404353d  }  // -0.0270016369910074
                        }),
                    HiddenBiases = new[] { 0d, 0d },
                    HiddenWeights = new MatrixD(
                        new[,]
                        {
                            {
                                -0.000828975875537991d,
                                -0.00265745714756122d
                            }, 
                            {
                                -0.00213165225138341d,
                                -0.00683346123658601d
                            }
                        }),
                };

                calculatedDeltas.OutputWeights.ShouldEqualByValue(expectedOutputDeltas.OutputWeights);
                calculatedDeltas.OutputBiases.ShouldEqualByValue(expectedOutputDeltas.OutputBiases);
                calculatedDeltas.HiddenWeights.ShouldEqualByValue(expectedOutputDeltas.HiddenWeights);
                calculatedDeltas.HiddenBiases.ShouldEqualByValue(expectedOutputDeltas.HiddenBiases);
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

                var exHiddenDelta00 = -0.0023685025015371172d;
                var exHiddenDelta01 = -0.0075927347073177845d;

                var expectedDeltas = new DeltasForNeuralNet
                {

                    OutputBiases = new[] { 0d, 0d, 0d },
                    OutputWeights = new MatrixD(
                        new[,]
                        {
                            {exOutputDelta1 *  net.HiddenLayer[0].FiringRate, exOutputDelta2 * net.HiddenLayer[0].FiringRate, exOutputDelta3 * net.HiddenLayer[0].FiringRate },
                            {exOutputDelta1 *  net.HiddenLayer[1].FiringRate, exOutputDelta2 * net.HiddenLayer[1].FiringRate, exOutputDelta3 * net.HiddenLayer[1].FiringRate }
                        }),
                    HiddenBiases = new[] { 0d, 0d },
                    HiddenWeights = new MatrixD(
                        new[,]
                        {
                            //TODO this is more complicated
                        {exHiddenDelta00 *  net.HiddenLayer[0].Inputs[0].Source.FiringRate, exHiddenDelta01 * net.HiddenLayer[1].Inputs[0].Source.FiringRate },
                        {exHiddenDelta00 *  net.HiddenLayer[0].Inputs[1].Source.FiringRate, exHiddenDelta01 * net.HiddenLayer[1].Inputs[1].Source.FiringRate }
                        }),
                };
                calculatedDeltas.OutputWeights.ShouldEqualByValue(expectedDeltas.OutputWeights);
                calculatedDeltas.OutputBiases.ShouldEqualByValue(expectedDeltas.OutputBiases);
                //WIP calculatedDeltas.HiddenWeights.ShouldEqualByValue(expectedDeltas.HiddenWeights);
                calculatedDeltas.HiddenBiases.ShouldEqualByValue(expectedDeltas.HiddenBiases);
            }

        }
    }
}