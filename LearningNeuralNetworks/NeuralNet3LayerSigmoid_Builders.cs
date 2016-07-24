using System;

namespace LearningNeuralNetworks
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
            var hiddenLength = (int)Math.Sqrt(inputToHiddenWeights.Length);
            var outputLength = hiddenToOutputWeights.Length / hiddenLength;
            var inputToHiddenWeightMatrix = new double[inputLength, hiddenLength];

            for (int r = 0; r < inputLength; r++)
            for (int c = 0; c < hiddenLength; c++)
            {
                inputToHiddenWeightMatrix[r, c] = inputToHiddenWeights[r * inputLength + c];
            }
            var hiddenToOutputWeightMatrix = new double[hiddenLength, outputLength];
            for (int r = 0; r < hiddenLength; r++)
            for (int c = 0; c < outputLength; c++)
            {
                hiddenToOutputWeightMatrix[r, c] = inputToHiddenWeights[r * inputLength + c];
            }
            return new NeuralNet3LayerSigmoid(inputToHiddenWeightMatrix, hiddenToOutputWeightMatrix,hiddenBiases, outputBiases);
        }
    }
}