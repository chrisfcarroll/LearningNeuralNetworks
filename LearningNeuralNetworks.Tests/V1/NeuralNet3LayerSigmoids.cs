using System;
using System.Linq;
using LearningNeuralNetworks.V1;
using NUnit.Framework;
using TestBase.Shoulds;

namespace LearningNeuralNetworks.Tests.V1
{
    [TestFixture]
    public partial class NeuralNet3LayerSigmoids
    {
        [TestFixture]
        public class OutputTheExpectedFunctionOfTheirInputsWeightsBiasesAndSigmoidActivation
        {
            [TestCase(1, 1, 1)]
            [TestCase(0, 0, 0)]
            [TestCase(0, 1, 1)]
            [TestCase(0, 0.5, 0)]
            [TestCase(0, 0, 0.5)]
            public void Given_Biases0_Weights0__ReturnsHalfHalfHalf__ForAnyInput(params double[] inputs)
            {
                var net = new NeuralNet3LayerSigmoid(3,1,3)
                            .SetBiases(new[] { 0d }, new [] {0d,0d,0d})
                            .SetInputToHiddenWeights( new double[3,1]{ {0}, { 0}, {0}} )
                            .SetHiddenToOutputWeights(new double[1,3] { { 0,0,0 } });
                //
                Console.WriteLine(net.OutputFor(inputs).ToArray());
                //
                net.OutputFor(inputs).ShouldEqualByValue(new ZeroToOne[]{0.5, 0.5, 0.5});
            }

            [TestCase(1, 1, 1)]
            [TestCase(0, 0, 0)]
            [TestCase(0, 1, 1)]
            [TestCase(0.5, 0.5, 0)]
            [TestCase(0.5, 0.5, 0.5)]
            public void Given__BiasesHigh_WeightsHigh__Returns111__ForAnyInput(params double[] inputs)
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
            public void Given__BiasesLow_WeightsLow__Returns000__ForAnyInput(params double[] inputs)
            {
                const double low = Neuron.Low;
                var net = new NeuralNet3LayerSigmoid(3, 1, 3)
                            .SetBiases(new[] { low }, new[] { low, low, low })
                            .SetInputToHiddenWeights(new double[3, 1] { { low }, { low }, { low } })
                            .SetHiddenToOutputWeights(new double[1, 3] { { low, low, low } });

                //
                Console.WriteLine(net.OutputFor(inputs));

                net.OutputFor(inputs).ToArray().ShouldBe(new ZeroToOne[] { 0,0,0 });
            }

            [TestCase(new[] { 0.35d, 0.9d }, new[] { 0.1d, 0.4d, 0.8d, 0.6d }, new[] { 0.3d, 0.9d }, 0.68026719669864855d, 0.66373869740435276d, 0.69028349290764435d)]
            public void Given__ExampleWeightsFor221Net__HasExpectedFiringRatesAndOutput(double[] inputs, double[] inputToHidden, double[] hiddenToOutput, double expectedH0, double expectedH1, double expectedOutput)
            {
                var net = NeuralNet3LayerSigmoid.FromFlatWeightArrays(2, inputToHidden, hiddenToOutput);
                //
                net.ActivateInputs(inputs);
                net.HiddenLayer[0].FiringRate.ShouldBe( expectedH0, "Hidden 0 Firing Rate");
                net.HiddenLayer[1].FiringRate.ShouldBe( expectedH1, "Hidden 1 Firing Rate");
                net.OutputLayer[0].FiringRate.ShouldBe( expectedOutput, "Output 0 Firing Rate");
                net.OutputFor(inputs).ShouldBe( new ZeroToOne[] { expectedOutput} );
            }
        }
    }
}