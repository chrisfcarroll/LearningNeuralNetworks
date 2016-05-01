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
            [TestCase(SigmoidNeuron.High, SigmoidNeuron.High, SigmoidNeuron.High)]
            [TestCase(0, 0, 0)]
            [TestCase(0, SigmoidNeuron.High, SigmoidNeuron.High)]
            [TestCase(-SigmoidNeuron.High, -SigmoidNeuron.High, 0)]
            [TestCase(-SigmoidNeuron.High, -SigmoidNeuron.High, -SigmoidNeuron.High)]
            public void Given_Biases0_Weights0__ReturnsHalfHalfHalf(params double[] inputs)
            {
                var net = new NeuralNet3LayerSigmoid(3,1,3)
                            .SetBiases(new[] { 0d, 0d, 0d }, new[] { 0d }, new [] {0d,0d,0d})
                            .SetInputToHiddenWeights( new double[3,1]{ {0}, { 0}, {0}} )
                            .SetHiddenToOutputWeights(new double[1,3] { { 0,0,0 } });
                //
                Console.WriteLine(net.OutputFor(inputs).ToArray());

                net.OutputFor(inputs).ShouldEqualByValue(new ZeroToOne[]{0.5, 0.5, 0.5});
            }

            [TestCase(SigmoidNeuron.High, SigmoidNeuron.High, SigmoidNeuron.High)]
            [TestCase(0, 0, 0)]
            [TestCase(0, SigmoidNeuron.High, SigmoidNeuron.High)]
            [TestCase(-SigmoidNeuron.High, -SigmoidNeuron.High, 0)]
            [TestCase(-SigmoidNeuron.High, -SigmoidNeuron.High, -SigmoidNeuron.High)]
            public void Given__BiasesHigh_WeightsHigh__Returns111(params double[] inputs)
            {
                const double h = SigmoidNeuron.High;
                var net = new NeuralNet3LayerSigmoid(new[,] { { h }, { h }, { h } }, new[,]{ { h,h,h } }, new[] { h, h, h }, new[] { h }, new[] { h, h, h });

                //
                Console.WriteLine(net.OutputFor(inputs));

                net.OutputFor(inputs).ToArray().ShouldBe(new ZeroToOne[] {1,1,1});
            }

            [TestCase(SigmoidNeuron.High, SigmoidNeuron.High, SigmoidNeuron.High)]
            [TestCase(0, 0, 0)]
            [TestCase(0, SigmoidNeuron.High, SigmoidNeuron.High)]
            [TestCase(-SigmoidNeuron.High, -SigmoidNeuron.High, 0)]
            [TestCase(-SigmoidNeuron.High, -SigmoidNeuron.High, -SigmoidNeuron.High)]
            public void Given__BiasesLow_WeightsLow__Returns000(params double[] inputs)
            {
                const double low = -SigmoidNeuron.High;
                var net = new NeuralNet3LayerSigmoid(3, 1, 3)
                            .SetBiases(new[] { low, low, low }, new[] { low }, new[] { low, low, low })
                            .SetInputToHiddenWeights(new double[3, 1] { { low }, { low }, { low } })
                            .SetHiddenToOutputWeights(new double[1, 3] { { low, low, low } });

                //
                Console.WriteLine(net.OutputFor(inputs));

                net.OutputFor(inputs).ToArray().ShouldBe(new ZeroToOne[] { 0,0,0 });
            }

        }
    }
}