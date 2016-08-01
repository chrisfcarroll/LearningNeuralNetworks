using System;
using System.Linq;
using LearningNeuralNetworks.LearningAlgorithms;
using LearningNeuralNetworks.Maths;
using LearningNeuralNetworks.V1;
using NUnit.Framework;
using TestBase.Shoulds;

namespace LearningNeuralNetworks.Tests
{
    [TestFixture]
    public class NNSigmoids
    {
        [Test]
        public void InitalisesWithRandomWeightsAndBiases()
        {
            var net = new NNSigmoid(new[] {10, 20, 13});
            net.Biases.ShouldAll(b =>
                                 {
                                     b.ByRows().Where(r=>r.Length>1).ShouldAll(
                                                          row =>
                                                          {
                                                              var avg = row.Average().ShouldBeBetween(0.1, 0.9);
                                                              row.Average(x => Math.Abs(x - avg)).ShouldBeBetween(0.1, 0.7);
                                                          });
                                 });
        }

        [TestCase(new[] { 0.35d, 0.9d }, new[] { 0.1d, 0.8d, 0.4d, 0.6d }, new[] { 0.3d, 0.9d }, 0.5d)]
        public void Given__221_SigmoidNetwork(double[] inputs, double[] inputToHiddenWeights, double[] hiddenToOutputWeights, double target)
        {
            var net = NNSigmoid.FromFlatWeightArrays(inputs.Length, inputToHiddenWeights, hiddenToOutputWeights);
            var input = new MatrixD(inputs.Length, 1, inputs);
            var targets = new MatrixD(1,1,target);
            var calculatedDeltas = net.backprop(input, targets);

            /*
             * 0.680267196698649d, 0.663738697404353d, 0.690283492907644d
             */
            var output = net.feedforward(input);
            var exOutputDelta = -0.040681125112339026d;
            var exHiddenDelta0 = -0.0023685025015371172d;
            var exHiddenDelta1 = -0.0075927347073177845d;

            var expectedDeltaValues = new
            {

                OutputBiases = new MatrixD(1,1,exOutputDelta),
                HiddenBiases = new MatrixD(2,1, new []{ -0.00260919207722715d, -0.00782757623168145d }),

                OutputWeights = new MatrixD(
                    new[,]
                    {
                            { -0.027674034938717864 }, // -0.040681125112339026d *  0.680267196698649d +/- last two d.p.s
                            { -0.027001636991007407 }  // -0.040681125112339026d *  0.663738697404353d +/- last two d.p.s
                    }),

                HiddenWeights = new MatrixD(
                    new[,]{
                            { -0.000913217227029503d, -0.0026574571475612243d },
                            { -0.0021316522513834054d, -0.0068334612365860059d }
                        }),
            };
            //
            calculatedDeltas.Item2.Last().ShouldEqualByValue(expectedDeltaValues.OutputBiases, "OutputBiases");
            calculatedDeltas.Item1.Last().ShouldEqualByValue(expectedDeltaValues.OutputWeights, "OutputWeights");
            calculatedDeltas.Item2.First().ShouldEqualByValue(expectedDeltaValues.HiddenBiases, "HiddenBiases");
            calculatedDeltas.Item1.First().ShouldEqualByValue(expectedDeltaValues.HiddenWeights, "HiddenWeights");
        }

        [TestCase(new[] { 0.35d, 0.9d }, new[] { 0.1d, 0.4d, /**/ 0.8d, 0.6d }, new[] { 0.3d, 0, 0,    /**/  0.9d, 0, 0 }, 0.5d)]
        [TestCase(new[] { 0.35d, 0.9d }, new[] { 0.1d, 0.4d, /**/ 0.8d, 0.6d }, new[] { 0.3d, 0, 0.3d, /**/  0.9d, 0, 0.9d }, 0.5d)]
        [TestCase(new[] { 0.35d, 0.9d }, new[] { 0.1d, 0.4d, /**/ 0.8d, 0.6d }, new[] { 0.3d, 0.3d, 0, /**/  0.9d, 0.9d, 0 }, 0.5d)]
        public void Given__223_SigmoidNetwork(double[] inputs, double[] inputToHiddenWeights, double[] hiddenToOutputWeights, double target)
        {
            var net = NNSigmoid.FromFlatWeightArrays(inputs.Length, inputToHiddenWeights, hiddenToOutputWeights);
            var input = new MatrixD(inputs.Length, 1, inputs);
            var targets = new MatrixD(3, 1, target);
            var calculatedDeltas = net.backprop(input, targets);

            var exOutputDelta1 = -0.040681125112339026d;
            var exOutputDelta2 = hiddenToOutputWeights[1].Equals(0) ? 0 : exOutputDelta1;
            var exOutputDelta3 = hiddenToOutputWeights[2].Equals(0) ? 0 : exOutputDelta1;

            var exHiddenDelta00 = -0.0023685025015371172d;
            var exHiddenDelta01 = -0.0075927347073177845d;

            //calculatedDeltas.Item1.First().ShouldEqualByValue(expectedDeltas.HiddenWeights);
            //calculatedDeltas.Item1.Last().ShouldEqualByValue(expectedDeltas.OutputWeights);
            calculatedDeltas.Item2.First().ShouldEqualByValue(new MatrixD(2,1,0));
            calculatedDeltas.Item2.Last().ShouldEqualByValue( new MatrixD(3,1,0));
        }

        [TestCase(new[] { 0.35d, 0.9d }, new[] { 0.1d, 0.4d, 0.8d, 0.6d, 0.1d, 0.4d, 0.8d, 0.6d }, new[] { 0.3d, 0.3d, 0, 0.9d, 0.9d, 0, 0.3d, 0.3d, 0, 0.9d, 0.9d, 0 }, 0.5d)]
        [TestCase(new[] { 0.35d, 0.9d, 0.3 }, new[] { 0.1d, 0.4d, 0.8d, 0.6d, 0.1d, 0.4d }, new[] { 0.3d, 0.3d, 0, 0.9d, 0.9d, 0, 0.3d, 0.3d, 0, 0.9d }, 0.5d)]
        public void CanCalculateBackProp__Given__LayersOfVariousSizes(double[] inputs, double[] inputToHiddenWeights, double[] hiddenToOutputWeights, double target)
        {
            var net = NNSigmoid.FromFlatWeightArrays(inputs.Length, inputToHiddenWeights, hiddenToOutputWeights);
            var input = new MatrixD(inputs.Length, 1, inputs);
            var outSize = hiddenToOutputWeights.Length * inputs.Length / inputToHiddenWeights.Length;
            var targets = new MatrixD(outSize, 1, target);
            var calculatedDeltas = net.backprop(input, targets);

        }
    }
}
