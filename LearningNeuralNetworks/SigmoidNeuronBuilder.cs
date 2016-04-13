using System.Collections.Generic;
using System.Linq;

namespace LearningNeuralNetworks
{
    public class SigmoidNeuronBuilder
    {
        public static SigmoidNeuron FixedSensorOn() { return new SigmoidNeuron { bias = SigmoidNeuron.High }; }
        public static SigmoidNeuron FixedSensorOff() { return new SigmoidNeuron { bias = -SigmoidNeuron.High }; }
        public static SigmoidNeuron FixedSensor(double fixedBias) { return new SigmoidNeuron { bias = fixedBias}; }

        public static SigmoidNeuron Nand(params SigmoidNeuron[] inputs) { return Nand((IEnumerable<SigmoidNeuron>) inputs); }

        public static SigmoidNeuron Nand(IEnumerable<SigmoidNeuron> inputs)
        {
            var num = inputs.Count();
            var nand = new SigmoidNeuron
            {
                bias = SigmoidNeuron.High * num,
                Inputs = inputs.Select(p => new Sinput(p, -SigmoidNeuron.High)).ToArray()
            };
            return nand;
        }

        public static SigmoidNeuron Or(params SigmoidNeuron[] inputs) { return Or((IEnumerable<SigmoidNeuron>)inputs); }

        public static SigmoidNeuron Or(IEnumerable<SigmoidNeuron> inputs)
        {
            return new SigmoidNeuron
            {
                bias = 0,
                Inputs = inputs.Select(p => new Sinput(p, SigmoidNeuron.High)).ToArray()
            };
        }

        public static explicit operator SigmoidNeuronBuilder(bool fixedSensorValue) { return new SigmoidNeuronBuilder(fixedSensorValue ? FixedSensorOn() : FixedSensorOff()); }
        public static explicit operator SigmoidNeuron(SigmoidNeuronBuilder builder) { return builder.builtInstance; }


        protected SigmoidNeuronBuilder(SigmoidNeuron builtInstance) { this.builtInstance = builtInstance; }
        readonly SigmoidNeuron builtInstance;
    }
}