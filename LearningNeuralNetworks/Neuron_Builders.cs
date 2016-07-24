using LearningNeuralNetworks.Maths;

namespace LearningNeuralNetworks
{
    public partial class Neuron
    {
        public Neuron(ActivationFunction activationFunction) { ActivationFunction = activationFunction; }

        public static Neuron NewSigmoid() { return new Neuron(MathExt.Sigmoid);}
        public static Neuron NewSensor() { return new Neuron(SensorNeuronBuilder.Id); }

        public static Neuron[] NewSigmoidArray(int layerSize)
        {
            var result = new Neuron[layerSize];
            for(int i=0; i<layerSize; i++){ result[i] = new Neuron(MathExt.Sigmoid); }
            return result;
        }
        public static Neuron[] NewSensorArray(int layerSize)
        {
            var result = new Neuron[layerSize];
            for (int i = 0; i < layerSize; i++) { result[i]= new Neuron(SensorNeuronBuilder.Id); }
            return result;
        }
    }
}