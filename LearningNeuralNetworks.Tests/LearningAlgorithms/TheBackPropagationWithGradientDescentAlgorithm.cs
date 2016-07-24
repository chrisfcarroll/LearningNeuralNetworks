using System;
using System.Linq;
using LearningNeuralNetworks.LearningAlgorithms;
using LearningNeuralNetworks.Maths;
using NUnit.Framework;
using TestBase.Shoulds;

namespace LearningNeuralNetworks.Tests.LearningAlgorithms
{
    [TestFixture, Ignore("WIP July 2016")]
    public class TheBackPropagationWithGradientDescentAlgorithm
    {
        [TestCase(new []{0.35d, 0.9d}, new[] {0.1d, 0.8d, 0.4d, 0.6d}, new[] { 0.3d, 0.9d }, 0.5d   )]
        public void CalculatesDeltaNablaOfErrorInSigmoidNetworkCorrectly(double[] inputs, double[] inputToHiddenWeights, double[] hiddenToOutputWeights, double target )
        {
            var net= NeuralNet3LayerSigmoid.FromFlatWeightArrays(inputs.Length, inputToHiddenWeights, hiddenToOutputWeights);

            var calculatedDeltas = BackPropagationWithGradientDescent.ErrorFor(net, new ZeroToOne[] {target});
            Console.WriteLine(calculatedDeltas);

            var exOutputDelta = -0.0406d;
            var exH1Delta = -2.406e-3d;
            var exH2Delta = -7.916e-3d;
            var expected= new DeltasFor2Layers
            {

                OutputBiases = new[] {0d},
                HiddenToOutput = new MatrixD(
                    new[,]
                    {
                        {exOutputDelta *  net.HiddenLayer[0].FiringRate }, 
                        {exOutputDelta *  net.HiddenLayer[1].FiringRate }
                    }),

                HiddenBiases = new[] { 0d, 0d },
                InputToHidden = new MatrixD(
                    new[,]
                    {
                        {exH1Delta * inputs[0], exH1Delta * inputs[1]},
                        {exH2Delta * inputs[0], exH2Delta * inputs[1] }
                    })
            };
            calculatedDeltas.ShouldEqualByValue(expected);
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