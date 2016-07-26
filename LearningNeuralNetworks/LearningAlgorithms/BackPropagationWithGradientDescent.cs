using System.Collections.Generic;
using System.Linq;
using LearningNeuralNetworks.Maths;

namespace LearningNeuralNetworks.LearningAlgorithms
{
    public class BackPropagationWithGradientDescent : LearningAlgorithm
    {
        //TODO trainingrate is ignored
        public override InterpretedNet<TData, TLabel> Apply<TData,TLabel>(InterpretedNet<TData, TLabel> net, IEnumerable<Pair<TData, TLabel>> trainingData, double trainingRateEta, int iterations=1)
        {
            for(int i=0; i<iterations; i++)
            foreach (var pair in trainingData)
            {
                var deltas = DeltasFor(net, pair);
                net.Net.DeltaBiases( deltas.HiddenBiases, deltas.OutputBiases);
                net.Net.DeltaHiddenToOutputWeights(deltas.OutputWeights);
                net.Net.DeltaInputToHiddenWeights(deltas.HiddenWeights);
            }
            return net;
        }

        public static DeltasFor2LayersOfNet DeltasFor<TData, TLabel>(InterpretedNet<TData, TLabel> net, Pair<TData,TLabel> target)
        {
            return DeltasFor(
                net.Net, 
                net.InputEncoding(target.Data).Select(i=>(double)i), 
                net.ReverseInterpretation(target.Label)
                );
        }

        public static DeltasFor2LayersOfNet DeltasFor(NeuralNet3LayerSigmoid net, IEnumerable<double> inputs, IEnumerable<ZeroToOne> targets)
        {
            var outputs = net.OutputFor(inputs).ToArray();
            var outputDeltas = outputs
                               .Zip(targets, (o, target) =>  (target - o) * o * (1 - o)) /* see e.g. wikipedia https://en.wikipedia.org/wiki/Backpropagation#Derivation */
                               .ToArray();

            var outputWeightDeltas = new MatrixD(net.HiddenToOutput.RowCount, net.HiddenToOutput.ColumnCount);
            var hiddenDeltaParts = new MatrixD(net.HiddenToOutput.RowCount, net.HiddenToOutput.ColumnCount);
            for (int i = 0; i < outputWeightDeltas.RowCount; i++)
            for (int j = 0; j < outputWeightDeltas.ColumnCount; j++)
            {
                outputWeightDeltas[i, j] = net.HiddenLayer[i].FiringRate * outputDeltas[j];
                hiddenDeltaParts[i, j] =  outputDeltas[j] * (net.HiddenToOutput[i,j] + outputWeightDeltas[i,j]) * outputs[j] * (1 - outputs[j]);
            }

            var hiddenDeltas = hiddenDeltaParts.ByRows().Select(row => row.Sum()).ToArray();
            var hiddenWeightDeltas = new MatrixD(net.InputToHidden.RowCount, net.InputToHidden.ColumnCount);
            for (int i = 0; i < hiddenWeightDeltas.RowCount; i++)
            for (int j = 0; j < hiddenWeightDeltas.ColumnCount; j++)
            {
                    hiddenWeightDeltas[i, j] = net.InputLayer[i].FiringRate * hiddenDeltas[j];
            }
            return new DeltasFor2LayersOfNet
            {
                OutputBiases = net.OutputLayer.Zip(outputDeltas, (neuron, delta) => neuron.Bias * delta).ToArray(),
                OutputWeights = outputWeightDeltas,
                HiddenBiases = net.HiddenLayer.Zip(hiddenDeltas, (neuron, delta) => neuron.Bias * delta).ToArray(),
                HiddenWeights = hiddenWeightDeltas
            };
        }
    }

    public class DeltasFor2LayersOfNet
    {
        public double[] OutputBiases;
        public double[] HiddenBiases;
        public MatrixD OutputWeights;
        public MatrixD HiddenWeights;

        public override string ToString()
        {
            var hiddenBiases = "[" + HiddenBiases?.Aggregate("", (s, b) => s + ", " + b) + "]";
            var outputBiases = "[" + OutputBiases?.Aggregate("", (s, b) => s + ", " + b) + "]";
            return
                $"HiddenBiases: {hiddenBiases ?? "none"}\n" +
                $"HiddenWeights: {OutputWeights}\n" +
                $"OutputBiases: {outputBiases ?? "none"}\n" +
                $"OutputWeights: {OutputWeights}\n" ;
        }
    }
}