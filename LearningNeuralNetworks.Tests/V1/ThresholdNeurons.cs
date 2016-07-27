using System.Linq;
using LearningNeuralNetworks.Tests.TestFrameworkChanges;
using LearningNeuralNetworks.V1;
using NUnit.Framework;
using TestBase.Shoulds;

namespace LearningNeuralNetworks.Tests.V1
{
    [TestFixture]
    public class ThresholdNeurons
    {
        [Test]
        public void Have_bias_and_have_inputs_with_weights()
        {
            var perceptron = new ThresholdNeuron();
            //
            perceptron.bias.ShouldCompile();
            perceptron.Inputs.ShouldCompile();
        }

        [TestCase(true,  "is the output of", true, false)]
        [TestCase(true,  "is the output of", true, true, true, false)]
        [TestCase(false, "is the output of", true, true)]
        [TestCase(false, "is the output of", true, true, true)]
        public void Can_model_Nand(bool expected, string descr, params bool[] fixedInputSensors)
        {
            var inputs = fixedInputSensors.Select(ThresholdNeuronBuilder.FixedSensor);
            //
            ThresholdNeuronBuilder.Nand(inputs).IsFiring.ShouldBe(expected);
        }

        [Test]
        public void Can_model_Or()
        {
            ThresholdNeuronBuilder.Or(ThresholdNeuronBuilder.FixedSensorOn(), ThresholdNeuronBuilder.FixedSensorOff()).IsFiring.ShouldBeTrue();
            ThresholdNeuronBuilder.Or(ThresholdNeuronBuilder.FixedSensorOff(), ThresholdNeuronBuilder.FixedSensorOff(),ThresholdNeuronBuilder.FixedSensorOn(), ThresholdNeuronBuilder.FixedSensorOff()).IsFiring.ShouldBeTrue();
            ThresholdNeuronBuilder.Or(ThresholdNeuronBuilder.FixedSensorOff(), ThresholdNeuronBuilder.FixedSensorOff()).IsFiring.ShouldBeFalse();
        }
    }
}
