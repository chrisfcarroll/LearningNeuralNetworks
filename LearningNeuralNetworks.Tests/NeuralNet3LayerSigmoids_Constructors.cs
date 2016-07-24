using System;
using System.Linq;
using System.Linq.Expressions;
using LearningNeuralNetworks.Maths;
using NUnit.Framework;
using TestBase.Shoulds;

namespace LearningNeuralNetworks.Tests
{
    [TestFixture]
    public partial class NeuralNet3LayerSigmoids
    {
        [TestFixture]
        public class FromFlatWeightArrays__Build3LayerNetsWithTheGivenWeightsAndBiases
        {
            [TestCase(new[] { 0.35d, 0.9d }, new[] { 0.1d, 0.4d, 0.8d, 0.6d }, new[] { 0.3d, 0.9d })]
            public void Given__a221Network(double[] inputs, double[] inputToHidden, double[] hiddenToOutput)
            {
                var net = NeuralNet3LayerSigmoid.FromFlatWeightArrays(inputs.Length, inputToHidden, hiddenToOutput);
                net.InputToHidden[0, 0].ShouldBe( inputToHidden[  0], "inputToHidden[0,0]");
                net.InputToHidden[0, 1].ShouldBe( inputToHidden[  1], "inputToHidden[0,1]");
                net.InputToHidden[1, 0].ShouldBe( inputToHidden[2+0], "inputToHidden[1,0]");
                net.InputToHidden[1, 1].ShouldBe( inputToHidden[2+1], "inputToHidden[1,1]");
                net.HiddenToOutput.ShouldEqualByValue( new MatrixD(new[,] { { hiddenToOutput[0] }, { hiddenToOutput[1] } }), "HiddenToOutput");
            }

            [TestCase(new[] { 0.1d, 0.2d, 0.3d }, new[] { 0.00d, 0.01d, 0.10d, 0.11d, 0.20d, 0.21d }, new[] { 0.00d, 0.01d, 0.02d, 0.10d, 0.11d, 0.12d })]
            public void Given__a323Network(double[] inputs, double[] inputToHidden, double[] hiddenToOutput)
            {
                var net = NeuralNet3LayerSigmoid.FromFlatWeightArrays(inputs.Length, inputToHidden, hiddenToOutput);
                net.InputToHidden[0, 0].ShouldBe(inputToHidden[    0]);
                net.InputToHidden[0, 1].ShouldBe(inputToHidden[    1]);
                net.InputToHidden[1, 0].ShouldBe(inputToHidden[2 + 0]);
                net.InputToHidden[1, 1].ShouldBe(inputToHidden[2 + 1]);
                net.InputToHidden[2, 0].ShouldBe(inputToHidden[4 + 0]);
                net.InputToHidden[2, 1].ShouldBe(inputToHidden[4 + 1]);

                net.HiddenToOutput[0, 0].ShouldBe(hiddenToOutput[    0]);
                net.HiddenToOutput[0, 1].ShouldBe(hiddenToOutput[    1]);
                net.HiddenToOutput[0, 2].ShouldBe(hiddenToOutput[    2]);
                net.HiddenToOutput[1, 0].ShouldBe(hiddenToOutput[3 + 0]);
                net.HiddenToOutput[1, 1].ShouldBe(hiddenToOutput[3 + 1]);
                net.HiddenToOutput[1, 2].ShouldBe(hiddenToOutput[3 + 2]);

                net.HiddenLayer.ShouldAll(n => n.Bias.ShouldBe(0));
                net.OutputLayer.ShouldAll(n => n.Bias.ShouldBe(0));
            }
        }

        [TestFixture]
        public class FromWeightAndBiasArrays__Build3LayerNetsWithTheGivenWeightsAndBiases
        {
            [TestCase(new[] { 0.35d, 0.9d }, new[] { 0.1d, 0.4d, 0.8d, 0.6d }, new[] { 0.3d, 0.9d }, new[]{0.1d, 0.2d}, new[]{0.01d})]
            public void Given__a221Network(double[] inputs, double[] inputToHidden, double[] hiddenToOutput, double[] hiddenBiases, double[] outputBiases)
            {
                var net = NeuralNet3LayerSigmoid.FromFlatArrays(inputs.Length, inputToHidden, hiddenToOutput, hiddenBiases, outputBiases);

                net.InputToHidden[0, 0].ShouldBe(inputToHidden[0], "inputToHidden[0,0]");
                net.InputToHidden[0, 1].ShouldBe(inputToHidden[1], "inputToHidden[0,1]");
                net.InputToHidden[1, 0].ShouldBe(inputToHidden[2 + 0], "inputToHidden[1,0]");
                net.InputToHidden[1, 1].ShouldBe(inputToHidden[2 + 1], "inputToHidden[1,1]");
                net.HiddenToOutput.ShouldEqualByValue(new MatrixD(new[,] { { hiddenToOutput[0] }, { hiddenToOutput[1] } }), "HiddenToOutput");

                net.HiddenLayer[0].Bias.ShouldBe(hiddenBiases[0], "hiddenBiases[0]");
                net.HiddenLayer[1].Bias.ShouldBe(hiddenBiases[1], "hiddenBiases[1]");
                net.OutputLayer[0].Bias.ShouldBe(outputBiases[0], "outputBiases[0]");

            }

            [TestCase(new[] { 0.1d, 0.2d, 0.3d }, new[] { 0.00d, 0.01d, 0.10d, 0.11d, 0.20d, 0.21d }, new[] { 0.00d, 0.01d, 0.02d, 0.10d, 0.11d, 0.12d }, new[] { 0.1d, 0.2d }, new[] { 0.01d, 0.02d, 0.03d })]
            public void Given__a323Network(double[] inputs, double[] inputToHidden, double[] hiddenToOutput, double[] hiddenBiases, double[] outputBiases)
            {
                var net = NeuralNet3LayerSigmoid.FromFlatArrays(inputs.Length, inputToHidden, hiddenToOutput, hiddenBiases, outputBiases);
                net.InputToHidden[0, 0].ShouldBe(inputToHidden[0]);
                net.InputToHidden[0, 1].ShouldBe(inputToHidden[1]);
                net.InputToHidden[1, 0].ShouldBe(inputToHidden[2 + 0]);
                net.InputToHidden[1, 1].ShouldBe(inputToHidden[2 + 1]);
                net.InputToHidden[2, 0].ShouldBe(inputToHidden[4 + 0]);
                net.InputToHidden[2, 1].ShouldBe(inputToHidden[4 + 1]);

                net.HiddenToOutput[0, 0].ShouldBe(hiddenToOutput[0]);
                net.HiddenToOutput[0, 1].ShouldBe(hiddenToOutput[1]);
                net.HiddenToOutput[0, 2].ShouldBe(hiddenToOutput[2]);
                net.HiddenToOutput[1, 0].ShouldBe(hiddenToOutput[3 + 0]);
                net.HiddenToOutput[1, 1].ShouldBe(hiddenToOutput[3 + 1]);
                net.HiddenToOutput[1, 2].ShouldBe(hiddenToOutput[3 + 2]);

                net.HiddenLayer[0].Bias.ShouldBe(hiddenBiases[0], "hiddenBiases[0]");
                net.HiddenLayer[1].Bias.ShouldBe(hiddenBiases[1], "hiddenBiases[1]");
                net.OutputLayer[0].Bias.ShouldBe(outputBiases[0], "outputBiases[0]");
                net.OutputLayer[1].Bias.ShouldBe(outputBiases[1], "outputBiases[1]");
                net.OutputLayer[2].Bias.ShouldBe(outputBiases[2], "outputBiases[2]");
            }
        }
    }
}