using System;
using System.Collections.Generic;

namespace LearningNeuralNetworks
{
    public delegate double ActivationFunction(double inputValue);

    /// <summary>
    /// Models a Sigmoid Neuron, see e.g. <see href="http://neuralnetworksanddeeplearning.com/chap1.html#sigmoid_neurons">this introduction</see>.
    /// </summary>
    public struct Neuron
    {
        public const double High = 1e100d;
        public const double Off = 0d;

        public double bias;
        public Sinput[] Inputs;
        public ActivationFunction ActivationFunction;

        public ZeroToOne FiringRate => ActivationFunction(Inputs.DotProduct() + bias);

        public static Neuron NewSigmoid() { return new Neuron {ActivationFunction = SigmoidNeuronExtensionMethods.Sigmoid};}
        public static Neuron NewSensor() { return new Neuron { ActivationFunction = x=>x }; }

        public static Neuron[] NewSigmoidArray(int layerSize)
        {
            var result = new Neuron[layerSize];
            for(int i=0; i<layerSize; i++){ result[i].ActivationFunction = SigmoidNeuronExtensionMethods.Sigmoid; }
            return result;
        }
        public static Neuron[] NewSensorArray(int layerSize)
        {
            var result = new Neuron[layerSize];
            for (int i = 0; i < layerSize; i++) { result[i].ActivationFunction = x=>x; }
            return result;
        }
    }

    ///<summary>Models a weighted input to a <see cref="Neuron"/> </summary>
    public struct Sinput
    {
        public double weight;
        public Neuron Source;
        public Sinput(Neuron source, double weight) : this()
        {
            this.weight = weight;
            this.Source = source;
        }
    }




	/// <summary>
	/// Models a double value in the range 0 &lt;= value &lt;= 1
	/// </summary>
  	public struct ZeroToOne
    {
        readonly double value;

        public ZeroToOne(double input)
        {
            if(input < 0 || input > 1) throw new ArgumentOutOfRangeException(nameof(input),"Must be between 0d and +1d");
            value = input;
        }

        public ZeroToOne(bool input) { value = input ? 1 : 0; }

        ///<returns>True if and only if this value > 0.5d</returns>
        public bool AsBool => value > 0.5d;

        public static implicit operator double(ZeroToOne input) { return input.value;}
        public static implicit operator ZeroToOne(double input) { return new ZeroToOne(input);}
        public static implicit operator bool(ZeroToOne input) { return input.AsBool;}
        public static implicit operator ZeroToOne(bool input) { return new ZeroToOne(input?1:0);}

        public override string ToString() { return value.ToString(); }
    }
}