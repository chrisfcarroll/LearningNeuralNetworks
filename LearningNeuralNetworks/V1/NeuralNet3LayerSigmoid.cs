using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using LearningNeuralNetworks.Maths;

namespace LearningNeuralNetworks.V1
{
    /// <summary>
    /// Models a 3 layer neural network (i.e. input, hidden and output layer) of <see cref="Neuron"/>
    /// and allows you to mutate all the weights and biases.
    /// </summary>
    public partial class NeuralNet3LayerSigmoid 
    {
        public Neuron[] InputLayer { get; }
        public Neuron[] HiddenLayer { get; }
        public Neuron[] OutputLayer { get; }
        public MatrixD InputToHidden { get; }
        public MatrixD HiddenToOutput { get; }

        /// <summary>
        /// Construct a 3 layer sigmoid neural net with all weights and biases zero.
        /// The net can be mutated with <see cref="SetInputToHiddenWeights"/>, <see cref="SetHiddenToOutputWeights"/> and <see cref="SetBiases"/>
        /// </summary>
        /// <param name="inputLayerSize"></param>
        /// <param name="hiddenLayerSize"></param>
        /// <param name="outputLayerSize"></param>
        public NeuralNet3LayerSigmoid(int inputLayerSize, int hiddenLayerSize, int outputLayerSize)
                               : this(
                                     new double[inputLayerSize, hiddenLayerSize], 
                                     new double[hiddenLayerSize, outputLayerSize],
                                     new double[hiddenLayerSize], 
                                     new double[outputLayerSize]
                                     ){}

        /// <summary>
        /// Construct a 3 layer sigmoid neural net with all weights and biases initialised to given values
        /// </summary>
        /// <param name="inputToHiddenWeights"></param>
        /// <param name="hiddenToOutputWeights"></param>
        /// <param name="hiddenLayerBiases"></param>
        /// <param name="outputLayerBiases"></param>
        public NeuralNet3LayerSigmoid(double[,] inputToHiddenWeights, double[,] hiddenToOutputWeights, double[] hiddenLayerBiases, double[] outputLayerBiases)
        {
            var inputLayerSize = inputToHiddenWeights.GetLength(0);
            var hiddenLayerSize = inputToHiddenWeights.GetLength(1);
            var outputLayerSize = hiddenToOutputWeights.GetLength(1);
            //
            Debug.Assert(hiddenLayerSize==hiddenToOutputWeights.GetLength(0), "Inconsistent matrix sizes for hidden layer" , "Inconsistent matrix sizes for hidden layer: inputTohidden {0}, hiddenToOutput {1}", hiddenLayerSize, hiddenToOutputWeights.GetLength(0));
            Debug.Assert(hiddenLayerSize == hiddenLayerBiases.Length, "Inconsistent sizes for hidden layer weight and biases", "Inconsistent sizes for hidden layer weights and biases: Weights {0}, Biases {1}", hiddenLayerSize, hiddenLayerBiases.Length);
            Debug.Assert(outputLayerSize == outputLayerBiases.Length, "Inconsistent sizes for output layer weight and biases", "Inconsistent sizes for output layer weights and biases: Weights {0}, Biases {1}", outputLayerSize, outputLayerBiases.Length);

            InputLayer = Neuron.NewSensorArray(inputLayerSize);
            HiddenLayer = Neuron.NewSigmoidArray(hiddenLayerSize);
            OutputLayer = Neuron.NewSigmoidArray(outputLayerSize);
            InputToHidden = new MatrixD(inputLayerSize, hiddenLayerSize);
            HiddenToOutput = new MatrixD(hiddenLayerSize, outputLayerSize);

            SetInputToHiddenWeights(inputToHiddenWeights);
            SetHiddenToOutputWeights(hiddenToOutputWeights);
            SetBiases(hiddenLayerBiases, outputLayerBiases);
        }

        public NeuralNet3LayerSigmoid SetInputToHiddenWeights(double[,] inputToHiddenWeights)
        {
            for (int j = 0; j < InputToHidden.ColumnCount; j++)
            {
                for (int i = 0; i < InputToHidden.RowCount; i++)
                {
                    InputToHidden[i, j] = inputToHiddenWeights[i, j];
                }
                HiddenLayer[j].Inputs = InputLayer.Select((n, ii) => new Ninput(n, InputToHidden[ii, j])).ToArray();
            }
            return this;
        }

        public NeuralNet3LayerSigmoid DeltaInputToHiddenWeights(MatrixD inputToHiddenWeights, double trainingRateEta=1)
        {
            for (int i = 0; i < InputToHidden.RowCount; i++)
            for (int j = 0; j < InputToHidden.ColumnCount; j++)
            {
                InputToHidden[i, j] += inputToHiddenWeights[i, j] * trainingRateEta;
                HiddenLayer[j].Inputs[i].Weight += inputToHiddenWeights[i, j] * trainingRateEta;
            }
            return this;

        }

        public NeuralNet3LayerSigmoid SetHiddenToOutputWeights(double[,] hiddenToOutputWeights)
        {
            for (int j = 0; j < HiddenToOutput.ColumnCount; j++)
            {
                for (int i = 0; i < HiddenToOutput.RowCount; i++)
                {
                    var Wij = HiddenToOutput[i, j] = hiddenToOutputWeights[i, j];
                }
                OutputLayer[j].Inputs = HiddenLayer.Select((n, ii) => new Ninput(n, HiddenToOutput[ii, j])).ToArray();
            }
            return this;
        }

        public NeuralNet3LayerSigmoid DeltaHiddenToOutputWeights(MatrixD hiddenToOutputWeights, double trainingRateEta=1)
        {
            for (int i = 0; i < HiddenToOutput.RowCount; i++)
            for (int j = 0; j < HiddenToOutput.ColumnCount; j++)
            {
                HiddenToOutput[i, j] += hiddenToOutputWeights[i, j] * trainingRateEta;
                OutputLayer[j].Inputs[i].Weight += hiddenToOutputWeights[i, j] * trainingRateEta;
            }
            return this;
        }

        public NeuralNet3LayerSigmoid ActivateInputs(IEnumerable<double> inputs)
        {
            var sharedLength = Math.Min(inputs.Count(), InputLayer.Length);
            for (int i = 0; i < sharedLength; i++)
            {
                InputLayer[i].Bias = inputs.ElementAt(i);
            }
            return this;
        }

        public NeuralNet3LayerSigmoid ActivateInputs(params double[] inputs)
        {
            ActivateInputs(inputs as IEnumerable<double>);
            return this;
        }

        public IEnumerable<ZeroToOne> OutputFor(params double[] inputs)
        {
            return ActivateInputs(inputs).LastOutputs.ToArray();
        }
        public IEnumerable<ZeroToOne> OutputFor(IEnumerable<double> inputs)
        {
            return ActivateInputs(inputs).LastOutputs.ToArray();
        }


        public IEnumerable<ZeroToOne> LastOutputs
        {
            get { return OutputLayer.Select(n => (ZeroToOne)n.FiringRate).ToArray(); }
        }

        public T LastOutputAs<T>(Func<IEnumerable<ZeroToOne>, T> interpretationOfOutputs)
        {
            return interpretationOfOutputs(LastOutputs);
        }

        public NeuralNet3LayerSigmoid SetBiases(double[] hiddenLayerBiases, double[] outputLayerBiases)
        {
            for (int i = 0; i < HiddenLayer.Length; i++){ HiddenLayer[i].Bias = hiddenLayerBiases[i]; }
            for (int i = 0; i < OutputLayer.Length; i++){ OutputLayer[i].Bias = outputLayerBiases[i]; }
            return this;
        }
        public NeuralNet3LayerSigmoid DeltaBiases(IEnumerable<double> hiddenLayerBiases = null, IEnumerable<double> outputLayerBiases = null, double trainingRateEta=1)
        {
            if(hiddenLayerBiases!=null) { for (int i = 0; i < HiddenLayer.Length; i++) { HiddenLayer[i].Bias += hiddenLayerBiases.ElementAt(i) * trainingRateEta; }}
            if(outputLayerBiases!=null) { for (int i = 0; i < OutputLayer.Length; i++) { OutputLayer[i].Bias += outputLayerBiases.ElementAt(i) * trainingRateEta; }}
            return this;
        }

        public override string ToString()
        {
            var sb= new StringBuilder( typeof(NeuralNet3LayerSigmoid).Name );
            sb.AppendLine($"<{InputLayer.Length},{HiddenLayer.Length},{OutputLayer.Length}>");
            sb.Append("Last Inputs   : ");
            sb.AppendLine(string.Join(",", (IEnumerable<Neuron>)InputLayer));
            sb.AppendLine("Input to Hidden Weights:");
            sb.Append(InputToHidden);
            sb.Append("Hidden Biases : ");
            sb.AppendLine(string.Join("," , HiddenLayer.Select(n=>n.Bias)));
            sb.AppendLine("Hidden to Output Weights:");
            sb.Append(HiddenToOutput);
            sb.Append("Output Biases : ");
            sb.AppendLine(string.Join(",", OutputLayer.Select(n=>n.Bias)));
            sb.AppendLine("Last Outputs : ");
            sb.Append(string.Join(",", LastOutputs));
            return sb.ToString();
        }
    }
}