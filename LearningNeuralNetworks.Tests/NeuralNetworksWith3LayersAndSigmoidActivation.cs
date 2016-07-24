using System;
using System.Linq;
using NUnit.Framework;
using TestBase.Shoulds;

namespace LearningNeuralNetworks.Tests
{
    [TestFixture]
    public class NeuralNetworksWith3LayersAndSigmoidActivation
    {
        [TestFixture]
        public class ProduceOutputsFromTheirInputsAsPerTheExpectedFunctionOfTheirWeightsBiasesAndSigmoidActivation
        {
            [TestCase(1, 1, 1)]
            [TestCase(0, 0, 0)]
            [TestCase(0, 1, 1)]
            [TestCase(0, 0.5, 0)]
            [TestCase(0, 0, 0.5)]
            public void Given_Biases0_Weights0__ReturnsHalfHalfHalf(params double[] inputs)
            {
                var net = new NeuralNet3LayerSigmoid(3,1,3)
                            .SetBiases(new[] { 0d }, new [] {0d,0d,0d})
                            .SetInputToHiddenWeights( new double[3,1]{ {0}, { 0}, {0}} )
                            .SetHiddenToOutputWeights(new double[1,3] { { 0,0,0 } });
                //
                Console.WriteLine(net.OutputFor(inputs).ToArray());

                net.OutputFor(inputs).ShouldEqualByValue(new ZeroToOne[]{0.5, 0.5, 0.5});
            }

            [TestCase(1, 1, 1)]
            [TestCase(0, 0, 0)]
            [TestCase(0, 1, 1)]
            [TestCase(0.5, 0.5, 0)]
            [TestCase(0.5, 0.5, 0.5)]
            public void Given__BiasesHigh_WeightsHigh__Returns111(params double[] inputs)
            {
                const double h = Neuron.High;
                var net = new NeuralNet3LayerSigmoid(new[,] { { h }, { h }, { h } }, new[,]{ { h,h,h } }, new[] { h }, new[] { h, h, h });

                //
                Console.WriteLine(net.OutputFor(inputs));

                net.OutputFor(inputs).ToArray().ShouldBe(new ZeroToOne[] {1,1,1});
            }

            [TestCase(1, 1, 1)]
            [TestCase(0, 0, 0)]
            [TestCase(0, 1, 1)]
            [TestCase(0.5, 0.5, 0)]
            [TestCase(0.5, 0.5, 0.5)]
            public void Given__BiasesLow_WeightsLow__Returns000(params double[] inputs)
            {
                const double low = -Neuron.High;
                var net = new NeuralNet3LayerSigmoid(3, 1, 3)
                            .SetBiases(new[] { low }, new[] { low, low, low })
                            .SetInputToHiddenWeights(new double[3, 1] { { low }, { low }, { low } })
                            .SetHiddenToOutputWeights(new double[1, 3] { { low, low, low } });

                //
                Console.WriteLine(net.OutputFor(inputs));

                net.OutputFor(inputs).ToArray().ShouldBe(new ZeroToOne[] { 0,0,0 });
            }

            [TestCase(new[] { 0.35d, 0.9d }, new[] { 0.1d, 0.4d, 0.8d, 0.6d }, new[] { 0.3d, 0.9d })]
            public void ConstructsExpectedNetworkFromFlatArrays(double[] inputs, double[] inputToHidden, double[] hiddenToOutput)
            {
                var net = NeuralNet3LayerSigmoid.FromFlatWeightArrays(2, inputToHidden, hiddenToOutput);
                net.InputToHidden[0, 0].ShouldBe(0.1d);
                net.InputToHidden[0, 1].ShouldBe(0.4d);
                net.InputToHidden[1, 0].ShouldBe(0.8d);
                net.InputToHidden[1, 1].ShouldBe(0.6d);
                net.HiddenToOutput.ShouldEqualByValue( new MatrixD(new[,] { { 0.3d }, { 0.9d } }), "HiddenToOutput");
            }

            [TestCase(new[] { 0.35d, 0.9d }, new[] { 0.1d, 0.4d, 0.8d, 0.6d }, new[] { 0.3d, 0.9d })]
            public void Given__a221NetworkWithExampleWeights_HasExpectedFiringRatesAndOutput(double[] inputs, double[] inputToHidden, double[] hiddenToOutput)
            {
                var net = NeuralNet3LayerSigmoid.FromFlatWeightArrays(2, inputToHidden, hiddenToOutput);
                var expectedOutput = 0.690283492907644d;
                //
                net.ActivateInputs(inputs);
                net.HiddenLayer[0].FiringRate.ShouldBe( (ZeroToOne)0.680267196698649d, "Hidden 0 Firing Rate");
                net.HiddenLayer[1].FiringRate.ShouldBe( (ZeroToOne)0.663738697404353d, "Hidden 1 Firing Rate");
                net.OutputLayer[0].FiringRate.ShouldBe( (ZeroToOne)expectedOutput, "Output 0 Firing Rate");
                net.OutputFor(inputs).ShouldBe( new ZeroToOne[] { expectedOutput} );
            }

        }
    }
}