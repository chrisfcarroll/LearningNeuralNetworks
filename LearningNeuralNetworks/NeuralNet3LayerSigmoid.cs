﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace LearningNeuralNetworks
{
    /// <summary>
    /// Models a 3 layer neural network (i.e. input, hidden and output layer) of <see cref="SigmoidNeuron"/>
    /// and allows you to mutate all the weights and biases.
    /// </summary>
    public class NeuralNet3LayerSigmoid 
    {
        public SigmoidNeuron[] InputLayer { get; }
        public SigmoidNeuron[] HiddenLayer { get; }
        public SigmoidNeuron[] OutputLayer { get; }
        public MatrixF InputToHidden { get; }
        public MatrixF HiddenToOutput { get; }

        /// <summary>
        /// Construct a 3 layer sigmoid neural net with all weights and biases zero.
        /// The net can be mutated with <see cref="SetInputToHiddenWeights"/>, <see cref="SetHiddenToOutputWeights"/> and <see cref="SetBiases"/>
        /// </summary>
        /// <param name="inputLayerSize"></param>
        /// <param name="hiddenLayerSize"></param>
        /// <param name="outputLayerSize"></param>
        public NeuralNet3LayerSigmoid(int inputLayerSize, int hiddenLayerSize, int outputLayerSize)
        {
            InputLayer= new SigmoidNeuron[inputLayerSize];
            HiddenLayer= new SigmoidNeuron[hiddenLayerSize];
            OutputLayer= new SigmoidNeuron[outputLayerSize];
            InputToHidden = new MatrixF(inputLayerSize, hiddenLayerSize);
            HiddenToOutput = new MatrixF(hiddenLayerSize, outputLayerSize);
        }

        /// <summary>
        /// Construct a 3 layer sigmoid neural net with all weights and biases initialised to given values
        /// </summary>
        /// <param name="inputToHiddenWeights"></param>
        /// <param name="hiddenToOutputWeights"></param>
        /// <param name="inputLayerBiases"></param>
        /// <param name="hiddenLayerBiases"></param>
        /// <param name="outputLayerBiases"></param>
        public NeuralNet3LayerSigmoid(double[,] inputToHiddenWeights, double[,] hiddenToOutputWeights, double[] inputLayerBiases, double[] hiddenLayerBiases, double[] outputLayerBiases)
        {
            var inputLayerSize = inputToHiddenWeights.GetLength(0);
            var hiddenLayerSize = inputToHiddenWeights.GetLength(1);
            var outputLayerSize = hiddenToOutputWeights.GetLength(1);
            //
            Debug.Assert(hiddenLayerSize==hiddenToOutputWeights.GetLength(0), "Inconsistent matrix sizes for hidden layer" , "Inconsistent matrix sizes for hidden layer: inputTohidden {0}, hiddenToOutput {1}", hiddenLayerSize, hiddenToOutputWeights.GetLength(0));
            Debug.Assert(inputLayerSize == inputLayerBiases.Length, "Inconsistent sizes for input layer weight and biases", "Inconsistent sizes for input layer weights and biases: Weights {0}, Biases {1}", inputLayerSize, inputLayerBiases.Length);
            Debug.Assert(hiddenLayerSize == hiddenLayerBiases.Length, "Inconsistent sizes for hidden layer weight and biases", "Inconsistent sizes for hidden layer weights and biases: Weights {0}, Biases {1}", hiddenLayerSize, hiddenLayerBiases.Length);
            Debug.Assert(outputLayerSize == outputLayerBiases.Length, "Inconsistent sizes for output layer weight and biases", "Inconsistent sizes for output layer weights and biases: Weights {0}, Biases {1}", outputLayerSize, outputLayerBiases.Length);

            InputLayer = new SigmoidNeuron[inputLayerSize];
            HiddenLayer = new SigmoidNeuron[hiddenLayerSize];
            OutputLayer = new SigmoidNeuron[outputLayerSize];
            InputToHidden = new MatrixF(inputLayerSize, hiddenLayerSize);
            HiddenToOutput = new MatrixF(hiddenLayerSize, outputLayerSize);
            SetInputToHiddenWeights(inputToHiddenWeights);
            SetHiddenToOutputWeights(hiddenToOutputWeights);
            SetBiases(inputLayerBiases, hiddenLayerBiases, outputLayerBiases);
        }

        public NeuralNet3LayerSigmoid SetInputToHiddenWeights(double[,] inputToHiddenWeights)
        {
            for (int i = 0; i < InputToHidden.RowCount; i++)
            for (int j = 0; j < InputToHidden.ColumnCount; j++)
            {
                InputToHidden[i, j] = inputToHiddenWeights[i, j];
            }
            return this;

        }
        public NeuralNet3LayerSigmoid DeltaInputToHiddenWeights(MatrixF inputToHiddenWeights)
        {
            for (int i = 0; i < InputToHidden.RowCount; i++)
            for (int j = 0; j < InputToHidden.ColumnCount; j++)
            {
                InputToHidden[i, j] += inputToHiddenWeights[i, j];
            }
            return this;

        }

        public NeuralNet3LayerSigmoid SetHiddenToOutputWeights(double[,] hiddenToOutputWeights)
        {
            for (int i = 0; i < HiddenToOutput.RowCount; i++)
            for (int j = 0; j < HiddenToOutput.ColumnCount; j++)
            {
                HiddenToOutput[i, j] = hiddenToOutputWeights[i, j];
            }
            return this;
        }

        public NeuralNet3LayerSigmoid DeltaHiddenToOutputWeights(MatrixF hiddenToOutputWeights)
        {
            for (int i = 0; i < HiddenToOutput.RowCount; i++)
            for (int j = 0; j < HiddenToOutput.ColumnCount; j++)
            {
                HiddenToOutput[i, j] += hiddenToOutputWeights[i, j];
            }
            return this;
        }

        public NeuralNet3LayerSigmoid ActivateInputs(IEnumerable<double> inputs)
        {
            var max = Math.Min(InputLayer.Length, inputs.Count());
            for (int i = 0; i < max; i++)
            {
                InputLayer[i].Inputs = new[] { new Sinput(SigmoidNeuronBuilder.FixedSensorOn(), inputs.Skip(i).First()) };
            }
            return this;
        }
        public NeuralNet3LayerSigmoid ActivateInputs(int[] inputs)
        {
            var max = Math.Min(InputLayer.Length, inputs.Count());
            for (int i = 0; i < max; i++)
            {
                InputLayer[i].Inputs = new[] { new Sinput(SigmoidNeuronBuilder.FixedSensorOn(), inputs[i]) };
            }
            return this;
        }
        public NeuralNet3LayerSigmoid ActivateInputs(byte[] inputs)
        {
            var max = Math.Min(InputLayer.Length, inputs.Count());
            for (int i = 0; i < max; i++)
            {
                InputLayer[i].Inputs = new[] { new Sinput(SigmoidNeuronBuilder.FixedSensorOn(), inputs[i]) };
            }
            return this;
        }

        public IEnumerable<ZeroToOne> OutputFor(params double[] inputs)
        {
            return ActivateInputs(inputs).LastOutputs.ToArray();
        }

        public IEnumerable<ZeroToOne> LastOutputs
        {
            get { return OutputLayer.Select(n => n.FiringRate).ToArray(); }
        }

        public T LastOutputAs<T>(Func<IEnumerable<ZeroToOne>, T> interpretationOfOutputs)
        {
            return interpretationOfOutputs(LastOutputs);
        }

        public NeuralNet3LayerSigmoid SetBiases(double[] inputLayerBiases, double[] hiddenLayerBiases, double[] outputLayerBiases)
        {
            for (int i = 0; i < InputLayer.Length; i++){ InputLayer[i].bias = inputLayerBiases[i];}
            for (int i = 0; i < HiddenLayer.Length; i++){ HiddenLayer[i].bias = hiddenLayerBiases[i]; }
            for (int i = 0; i < OutputLayer.Length; i++){ OutputLayer[i].bias = outputLayerBiases[i]; }
            return this;
        }
        public NeuralNet3LayerSigmoid DeltaBiases(IEnumerable<double> inputLayerBiases, IEnumerable<double> hiddenLayerBiases, IEnumerable<double> outputLayerBiases)
        {
            for (int i = 0; i < InputLayer.Length; i++) { InputLayer[i].bias += inputLayerBiases.ElementAt(i); }
            for (int i = 0; i < HiddenLayer.Length; i++) { HiddenLayer[i].bias += hiddenLayerBiases.ElementAt(i); }
            for (int i = 0; i < OutputLayer.Length; i++) { OutputLayer[i].bias += outputLayerBiases.ElementAt(i); }
            return this;
        }

    }
}