using System;

namespace LearningNeuralNetworks.V1
{
    public partial class NeuralNet3LayerSigmoid
    {
        /// <summary>
        /// Useful for creating networks from constants in e.g. Attribute declarations where only 1D arrays are allowed.
        /// this overload initialises all biases to zero.
        /// </summary>
        public static NeuralNet3LayerSigmoid FromFlatWeightArrays(int inputLength, double[] inputToHiddenWeights, double[] hiddenToOutputWeights)
        {
            var hiddenLength = (int)Math.Sqrt(inputToHiddenWeights.Length);
            var outputLength = hiddenToOutputWeights.Length / hiddenLength;
            var inputToHiddenWeightMatrix = new double[inputLength, hiddenLength];
            for (int r = 0; r < inputLength; r++)
            for (int c = 0; c < hiddenLength; c++)
            {
                inputToHiddenWeightMatrix[r, c] = inputToHiddenWeights[r * hiddenLength + c];
            }
            var hiddenToOutputWeightMatrix = new double[hiddenLength, outputLength];
            for (int r = 0; r < hiddenLength; r++)
            for (int c = 0; c < outputLength; c++)
            {
                hiddenToOutputWeightMatrix[r, c] = hiddenToOutputWeights[r * outputLength + c];
            }
            return new NeuralNet3LayerSigmoid(inputToHiddenWeightMatrix, hiddenToOutputWeightMatrix, new double[hiddenLength], new double[outputLength]);
        }

        /// <summary>Useful for creating networks from constants in e.g. Attribute declarations where only 1D arrays are allowed.</summary>
        public static NeuralNet3LayerSigmoid FromFlatArrays(int inputLength, 
                                                            double[] inputToHiddenWeights, 
                                                            double[] hiddenToOutputWeights,
                                                            double[] hiddenBiases,
                                                            double[] outputBiases)
        {
            return FromFlatWeightArrays(inputLength, inputToHiddenWeights, hiddenToOutputWeights)
                    .SetBiases(hiddenBiases, outputBiases);
        }



        public NeuralNet3LayerSigmoid Randomize(double scale=1)
        {
            var rnd= new Random();
            for (int c = 0; c < HiddenLayer.Length; c++)
            {
                for (int r = 0; r < InputLayer.Length; r++)
                {
                    HiddenLayer[c].Inputs[r].Weight = InputToHidden[r, c] = (rnd.NextDouble() - 0.5) * scale;
                }
                HiddenLayer[c].Bias = (rnd.NextDouble() - 0.5) * scale;
            }
            for (int c = 0; c < OutputLayer.Length; c++)
            {
                for (int r = 0; r < HiddenLayer.Length; r++)
                {
                    OutputLayer[c].Inputs[r].Weight = HiddenToOutput[r, c] = (rnd.NextDouble() - 0.5) * scale;
                }
                OutputLayer[c].Bias = (rnd.NextDouble() - 0.5) * scale;
            }
            return this;
        }
    }
}