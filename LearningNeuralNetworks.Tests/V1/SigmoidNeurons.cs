using LearningNeuralNetworks.Maths;
using LearningNeuralNetworks.Tests.TestFrameworkChanges;
using LearningNeuralNetworks.V1;
using NUnit.Framework;
using TestBase.Shoulds;

namespace LearningNeuralNetworks.Tests.V1
{
    [TestFixture]
    public class SigmoidNeurons
    {
        [Test]
        public void Have_bias_and_have_inputs_with_weights()
        {
            var sigmoidNeuron = Neuron.NewSigmoid();
            //
            sigmoidNeuron.Bias.ShouldCompile();
            sigmoidNeuron.Inputs.ShouldCompile();
        }

        [TestCase(true,  "is the binary output of", 1, 0)]
        [TestCase(true,  "is the binary output of", 1, 1, 1, 0)]
        [TestCase(false, "is the binary output of", 1, 1)]
        [TestCase(false, "is the binary output of", 1, 1, 1, 1)]
        public void Can_model_binary_Nand(bool expected, string descr, params double[] fixedInputSensors)
        {
            var inputs = SensorNeuronBuilder.WithValues(fixedInputSensors);
            //
          	bool isOn = SigmoidNeuronBuilder.Nand(inputs).FiringRate.AsSigmoidToBool();
            //
            isOn.ShouldBe(expected);
        }

        [Test]
        public void Can_model_binary_Or()
        {
            SigmoidNeuronBuilder.Or(SensorNeuronBuilder.On(), SensorNeuronBuilder.Off())
              	.FiringRate.AsSigmoidToBool()
                .ShouldBeTrue();

            SigmoidNeuronBuilder.Or(SensorNeuronBuilder.Off(), SensorNeuronBuilder.Off(),SensorNeuronBuilder.On(), SensorNeuronBuilder.Off())
              	.FiringRate.AsSigmoidToBool()
                .ShouldBeTrue();

            SigmoidNeuronBuilder.Or(SensorNeuronBuilder.Off(), SensorNeuronBuilder.Off())
              	.FiringRate.AsSigmoidToBool()
                .ShouldBeFalse();
        }
    }
}
