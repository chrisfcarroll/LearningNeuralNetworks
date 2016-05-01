using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NUnit.Framework;
using TestBase.Shoulds;

namespace LearningNeuralNetworks.Tests
{
    [TestFixture]
    public class TheMnistLearner1Builder
    {
        [Test]
        public void BuildsANetworkWith784InputNeurons()
        {
            var net= MnistSigmoidLearner1NetBuilder.Build(15);
            net.InputLayer.Length.ShouldBe(MnistParser.Image.ByteSize);
            net.HiddenLayer.Length.ShouldBe(15);
            net.OutputLayer.Length.ShouldBe(10);
        }
    }
}
