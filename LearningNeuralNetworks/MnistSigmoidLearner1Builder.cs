using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningNeuralNetworks
{
    public static class MnistSigmoidLearner1Builder
    {
        public static NeuralNet Build(int hiddenLayerSize= 15)
        {
            return new NeuralNet(784, hiddenLayerSize, 10);
        }
    }

    public class NeuralNet 
    {
        public SigmoidNeuron[] InputLayer { get; set; }
        public SigmoidNeuron[] HiddenLayer { get; set; }
        public SigmoidNeuron[] OutputLayer { get; set; }

        public NeuralNet(int inputLayerSize, int hiddenLayerSize, int outputLayerSize)
        {
            InputLayer= new SigmoidNeuron[inputLayerSize];
            HiddenLayer= new SigmoidNeuron[hiddenLayerSize];
            OutputLayer= new SigmoidNeuron[outputLayerSize];
        }
    }

}
