using System.Collections.Generic;
using System.Linq;

namespace LearningNeuralNetworks
{
    public class PerceptronBuilder
    {
        public static Perceptron FixedSensorOn() { return new Perceptron { bias = 1 }; }
        public static Perceptron FixedSensorOff() { return new Perceptron { bias = 0 }; }
        public static Perceptron FixedSensor(bool isOn) { return new Perceptron { bias = isOn ? 1 : 0 }; }

        public static Perceptron Nand(params Perceptron[] inputs) { return Nand((IEnumerable<Perceptron>) inputs); }

        public static Perceptron Nand(IEnumerable<Perceptron> inputs)
        {
            var num = inputs.Count();
            var nand = new Perceptron
            {
                bias = num,
                Inputs = inputs.Select(p => new Pinput(p, -1)).ToArray()
            };
            return nand;
        }

        public static Perceptron Or(params Perceptron[] inputs) { return Or((IEnumerable<Perceptron>)inputs); }

        public static Perceptron Or(IEnumerable<Perceptron> inputs)
        {
            return new Perceptron
            {
                bias = 0,
                Inputs = inputs.Select(p => new Pinput(p, 1)).ToArray()
            };
        }

        public static explicit operator PerceptronBuilder(bool fixedSensorValue) { return new PerceptronBuilder(fixedSensorValue ? FixedSensorOn() : FixedSensorOff()); }
        public static explicit operator Perceptron(PerceptronBuilder builder) { return builder.builtInstance; }


        protected PerceptronBuilder(Perceptron builtInstance) { this.builtInstance = builtInstance; }
        readonly Perceptron builtInstance;
    }
}