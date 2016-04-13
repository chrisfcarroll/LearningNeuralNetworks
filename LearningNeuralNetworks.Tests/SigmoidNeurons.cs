using System.Linq;
using LearningNeuralNetworks.Tests.TestFrameworkChanges;
using NUnit.Framework;
using TestBase.Shoulds;

namespace LearningNeuralNetworks.Tests
{
    [TestFixture]
    public class SigmoidNeurons
    {
        [Test]
        public void Have_bias_and_have_inputs_with_weights()
        {
            var sigmoidNeuron = new SigmoidNeuron();
            //
            sigmoidNeuron.bias.ShouldCompile();
            sigmoidNeuron.Inputs.ShouldCompile();
        }

        [TestCase(1d,  "is the output of", SigmoidNeuron.High, 0)]
        [TestCase(1d,  "is the output of", SigmoidNeuron.High, SigmoidNeuron.High, SigmoidNeuron.High, 0)]
        [TestCase(0d, "is the output of", SigmoidNeuron.High, SigmoidNeuron.High)]
        [TestCase(0d, "is the output of", SigmoidNeuron.High, SigmoidNeuron.High, SigmoidNeuron.High, SigmoidNeuron.High)]
        [Test, Ignore("Exploring")]
        public void Can_model_Nand(double expected, string descr, params double[] fixedInputSensors)
        {
            var inputs = fixedInputSensors.Select(SigmoidNeuronBuilder.FixedSensor);
            //
            var result = SigmoidNeuronBuilder.Nand(inputs);
            //
            result.FiringRate.ShouldBe( (ZeroToOne)expected);
        }

        [Test, Ignore("Exploring")]
        public void Can_model_Or()
        {
            SigmoidNeuronBuilder.Or(SigmoidNeuronBuilder.FixedSensorOn(), SigmoidNeuronBuilder.FixedSensorOff()).FiringRate.ShouldBe(1);
            SigmoidNeuronBuilder.Or(SigmoidNeuronBuilder.FixedSensorOff(), SigmoidNeuronBuilder.FixedSensorOff(),SigmoidNeuronBuilder.FixedSensorOn(), SigmoidNeuronBuilder.FixedSensorOff()).FiringRate.ShouldBe(1);

            var result = SigmoidNeuronBuilder.Or(SigmoidNeuronBuilder.FixedSensorOff(), SigmoidNeuronBuilder.FixedSensorOff());
            result.FiringRate.ShouldBe(0);
        }
    }
}
