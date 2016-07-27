using System.Collections.Generic;
using System.Linq;

namespace LearningNeuralNetworks.V1
{
    public class ThresholdNeuronBuilder
    {
        public static ThresholdNeuron FixedSensorOn() { return new ThresholdNeuron { bias = 1 }; }
        public static ThresholdNeuron FixedSensorOff() { return new ThresholdNeuron { bias = 0 }; }
        public static ThresholdNeuron FixedSensor(bool isOn) { return new ThresholdNeuron { bias = isOn ? 1 : 0 }; }

        public static ThresholdNeuron Nand(params ThresholdNeuron[] inputs) { return Nand((IEnumerable<ThresholdNeuron>) inputs); }

        public static ThresholdNeuron Nand(IEnumerable<ThresholdNeuron> inputs)
        {
            var num = inputs.Count();
            var nand = new ThresholdNeuron
            {
                bias = num,
                Inputs = inputs.Select(p => new ThresholdNeuron.Input(p, -1)).ToArray()
            };
            return nand;
        }

        public static ThresholdNeuron Or(params ThresholdNeuron[] inputs) { return Or((IEnumerable<ThresholdNeuron>)inputs); }

        public static ThresholdNeuron Or(IEnumerable<ThresholdNeuron> inputs)
        {
            return new ThresholdNeuron
            {
                bias = 0,
                Inputs = inputs.Select(p => new ThresholdNeuron.Input(p, 1)).ToArray()
            };
        }

        public static explicit operator ThresholdNeuronBuilder(bool fixedSensorValue) { return new ThresholdNeuronBuilder(fixedSensorValue ? FixedSensorOn() : FixedSensorOff()); }
        public static explicit operator ThresholdNeuron(ThresholdNeuronBuilder builder) { return builder.builtInstance; }


        protected ThresholdNeuronBuilder(ThresholdNeuron builtInstance) { this.builtInstance = builtInstance; }
        readonly ThresholdNeuron builtInstance;
    }
}