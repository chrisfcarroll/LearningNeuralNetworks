using System.Linq;
using LearningNeuralNetworks.Tests.TestFrameworkChanges;
using NUnit.Framework;
using TestBase.Shoulds;

namespace LearningNeuralNetworks.Tests
{
    [TestFixture]
    public class Perceptrons
    {
        [Test]
        public void Have_bias_and_have_inputs_with_weights()
        {
            var perceptron = new Perceptron();
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
            var inputs = fixedInputSensors.Select(PerceptronBuilder.FixedSensor);
            //
            PerceptronBuilder.Nand(inputs).IsFiring.ShouldBe(expected);
        }

        [Test]
        public void Can_model_Or()
        {
            PerceptronBuilder.Or(PerceptronBuilder.FixedSensorOn(), PerceptronBuilder.FixedSensorOff()).IsFiring.ShouldBeTrue();
            PerceptronBuilder.Or(PerceptronBuilder.FixedSensorOff(), PerceptronBuilder.FixedSensorOff(),PerceptronBuilder.FixedSensorOn(), PerceptronBuilder.FixedSensorOff()).IsFiring.ShouldBeTrue();
            PerceptronBuilder.Or(PerceptronBuilder.FixedSensorOff(), PerceptronBuilder.FixedSensorOff()).IsFiring.ShouldBeFalse();
        }
    }
}
