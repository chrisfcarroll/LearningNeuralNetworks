using System;
using System.Linq;
using LearningNeuralNetworks.Maths;
using LearningNeuralNetworks.V1;
using MnistParser;

namespace LearningNeuralNetworks
{
    public static class MnistLearnerSigmoidNetBuilder
    {
        public static InterpretedNet<Image,byte> Build(int hiddenLayerSize)
        {
            return new InterpretedNet<Image,byte>(
                        new NeuralNet3LayerSigmoid(784, hiddenLayerSize, 10),
                        image => image.As1Ddoubles.ToArray(),
                        e=>(byte) e.ArgMaxIndex(n=>n),
                        b => Enumerable.Range(0,10).Select(i => i==b? 1 :0).Select(i => (ZeroToOne)i),
                        (l,r) => Enumerable.Range(0, 10).Select(i => Math.Abs(l.ElementAt(i) - r.ElementAt(i)))
                        ) ;
        }
    }
}
