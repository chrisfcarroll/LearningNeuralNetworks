using System;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace LearningNeuralNetworks
{
    public static class MnistSigmoidLearner1NetBuilder
    {
        public static NeuralNet3LayerSigmoid Build(int hiddenLayerSize)
        {
            return new NeuralNet3LayerSigmoid(784, hiddenLayerSize, 10);
        }
    }

}
