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
            MnistSigmoidLearner1Builder.Build()
                .InputLayer
                .Length.ShouldBe(MnistParser.Image.ByteSize);
        }
    }
}
