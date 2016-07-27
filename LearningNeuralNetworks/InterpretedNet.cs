using System;
using System.Collections.Generic;
using System.Linq;
using LearningNeuralNetworks.LearningAlgorithms;
using LearningNeuralNetworks.Maths;
using LearningNeuralNetworks.V1;

namespace LearningNeuralNetworks
{
    public delegate double Distance(IEnumerable<ZeroToOne> left, IEnumerable<ZeroToOne> right);

    public delegate IEnumerable<double> Distances(IEnumerable<ZeroToOne> left, IEnumerable<ZeroToOne> right);

    /// <summary>
    /// Wraps a neural net with functions to translate:
    /// <list type="bullets">
    /// <item>Data to input layer weights</item>
    /// <item>Output layer weights to a Label</item>
    /// <item>A Label back to output layer weights</item>
    /// </list>
    /// And, a distance function for output layer weights. Which can also tell you which of two output weights is closer to a given label.
    /// </summary>
    /// <typeparam name="TData"></typeparam>
    /// <typeparam name="TLabel"></typeparam>
    public class InterpretedNet<TData,TLabel>
    {
        public static Distance PythagoreanDistance = (left, right) =>  left.Select( (l,i) =>  Math.Pow(l-right.ElementAt(i), 2) ).Sum().Sqrt();

        public Func<IEnumerable<ZeroToOne>, TLabel> OutputInterpretation { get; set; }
        public Func<TLabel, IEnumerable<ZeroToOne>> ReverseInterpretation { get; set; }
        public Func<TData,double[]>  InputEncoding { get; set; }
        public Distances Distances { get; set; }
        public Distance  Distance { get; set; }
        public NeuralNet3LayerSigmoid Net { get; }

        public TLabel LastOutput => Net.LastOutputAs(OutputInterpretation);

        public TLabel OutputFor(TData input) { return ActivateInputs(input).LastOutput; }

        public TLabel OutputFor(double[] input) { Net.ActivateInputs(input); return LastOutput; }

        public InterpretedNet(NeuralNet3LayerSigmoid net, Func<TData, double[]> inputEncoding, Func<IEnumerable<ZeroToOne>, TLabel> outputInterpretation, Func<TLabel, IEnumerable<ZeroToOne>> reverseInterpretation, Distances distancesFunction)
        {
            InputEncoding = inputEncoding;
            OutputInterpretation = outputInterpretation;
            ReverseInterpretation = reverseInterpretation;
            Distances = distancesFunction;
            Distance = PythagoreanDistance;
            Net = net;
        }

        public InterpretedNet<TData,TLabel> ActivateInputs(TData input)
        {
            Net.ActivateInputs(InputEncoding(input));
            return this;
        }
    }


    public static class InterpretedNetExtensions
    {
        public static IEnumerable<ZeroToOne> CloserOutputToTarget<TData,TLabel>(this InterpretedNet<TData,TLabel> @this, TLabel targetLabel, IEnumerable<ZeroToOne> candidateOutput1, IEnumerable<ZeroToOne> candidateOutput2)
        {
            var target = @this.ReverseInterpretation(targetLabel);
            return @this.Distance(target, candidateOutput1) < @this.Distance(target, candidateOutput2) ? candidateOutput1 : candidateOutput2;
        }

    }
}