using LearningNeuralNetworks.Maths;

namespace LearningNeuralNetworks.V1
{
    public partial class Neuron
    {
        public Neuron(ActivationFunction activationFunction) { ActivationFunction = activationFunction; }

        /// <summary>The Identity Function, x => x.</summary>
        public static ActivationFunction Id = x => x;

        /// <summary>The Logistic Function, x => 1 / (1 + e^-x)</summary>
        public static ActivationFunction Sigmoid = MathExt.Sigmoid;

        public static Neuron NewSigmoid() { return new Neuron(Sigmoid);}
        public static Neuron NewSensor() { return new Neuron(Id); }

        public static Neuron[] NewSigmoidArray(int layerSize)
        {
            var result = new Neuron[layerSize];
            for(int i=0; i<layerSize; i++){ result[i] = new Neuron(Sigmoid); }
            return result;
        }
        public static Neuron[] NewSensorArray(int layerSize)
        {
            var result = new Neuron[layerSize];
            for (int i = 0; i < layerSize; i++) { result[i]= new Neuron(Id); }
            return result;
        }
    }
}