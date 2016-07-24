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
            var sigmoidNeuron = new Neuron();
            //
            sigmoidNeuron.bias.ShouldCompile();
            sigmoidNeuron.Inputs.ShouldCompile();
        }

        [TestCase(true,  "is the binary output of", 1, 0)]
        [TestCase(true,  "is the binary output of", 1, 1, 1, 0)]
        [TestCase(false, "is the binary output of", 1, 1)]
        [TestCase(false, "is the binary output of", 1, 1, 1, 1)]
        public void Can_model_binary_Nand(bool expected, string descr, params double[] fixedInputSensors)
        {
            var inputs = fixedInputSensors.Select(InputNeuronBuilder.FixedSensor);
            //
          	bool isOn = SigmoidNeuronBuilder.Nand(inputs).FiringRate;
            //
            isOn.ShouldBe(expected);
        }

        [Test]
        public void Can_model_binary_Or()
        {
            SigmoidNeuronBuilder.Or(InputNeuronBuilder.FixedSensorOn(), InputNeuronBuilder.FixedSensorOff())
              	.FiringRate.AsBool
              	.ShouldBeTrue();

            SigmoidNeuronBuilder.Or(InputNeuronBuilder.FixedSensorOff(), InputNeuronBuilder.FixedSensorOff(),InputNeuronBuilder.FixedSensorOn(), InputNeuronBuilder.FixedSensorOff())
              	.FiringRate.AsBool
              	.ShouldBeTrue();

            SigmoidNeuronBuilder.Or(InputNeuronBuilder.FixedSensorOff(), InputNeuronBuilder.FixedSensorOff())
              	.FiringRate.AsBool
              	.ShouldBeFalse();
        }
    }
}
