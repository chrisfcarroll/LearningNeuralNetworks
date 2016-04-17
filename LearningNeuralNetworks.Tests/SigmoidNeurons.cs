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

        [TestCase(true,  "is the binary output of", SigmoidNeuron.High, 0)]
        [TestCase(true,  "is the binary output of", SigmoidNeuron.High, SigmoidNeuron.High, SigmoidNeuron.High, 0)]
        [TestCase(false, "is the binary output of", SigmoidNeuron.High, SigmoidNeuron.High)]
        [TestCase(false, "is the binary output of", SigmoidNeuron.High, SigmoidNeuron.High, SigmoidNeuron.High, SigmoidNeuron.High)]
        public void Can_model_binary_Nand(bool expected, string descr, params double[] fixedInputSensors)
        {
            var inputs = fixedInputSensors.Select(SigmoidNeuronBuilder.FixedSensor);
            //
          	bool isOn = SigmoidNeuronBuilder.Nand(inputs).FiringRate;
            //
            isOn.ShouldBe(expected);
        }

        [Test]
        public void Can_model_binary_Or()
        {
            SigmoidNeuronBuilder.Or(SigmoidNeuronBuilder.FixedSensorOn(), SigmoidNeuronBuilder.FixedSensorOff())
              	.FiringRate.AsBool
              	.ShouldBeTrue();

            SigmoidNeuronBuilder.Or(SigmoidNeuronBuilder.FixedSensorOff(), SigmoidNeuronBuilder.FixedSensorOff(),SigmoidNeuronBuilder.FixedSensorOn(), SigmoidNeuronBuilder.FixedSensorOff())
              	.FiringRate.AsBool
              	.ShouldBeTrue();

            SigmoidNeuronBuilder.Or(SigmoidNeuronBuilder.FixedSensorOff(), SigmoidNeuronBuilder.FixedSensorOff())
              	.FiringRate.AsBool
              	.ShouldBeFalse();
        }
    }
}
