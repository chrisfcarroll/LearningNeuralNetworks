using System;
using System.Collections.Generic;
using System.Linq;

namespace LearningNeuralNetworks.Maths
{
    public static class MathExt
    {
        public static bool AsSigmoidToBool(this double input){ return input >= 0.5; }
        public static bool As0To1(this double input) { return (ZeroToOne)input; }

        public static double DotProduct(this IEnumerable<Ninput> inputs)
        {
            return inputs == null ?
                0
                : inputs.Sum(i => i.Source.FiringRate * i.Weight);
        }

        public static double Sigmoid(this double input)
        {
            return 1d / (1d + Math.Exp(-input));
        }

        public static double SigmoidDerivative(this double input)
        {
            var sigmoid = Sigmoid(input);
            return  sigmoid * (1 - sigmoid);
        }

        public static double Sigmoid(this ZeroToOne input)
        {
            return 1d / (1d + Math.Exp(-input));
        }

        public static double SigmoidDerivative(this ZeroToOne input)
        {
            var sigmoid = Sigmoid(input);
            return sigmoid * (1 - sigmoid);
        }

        public static double Sqrt(this double input)
        {
            return Math.Sqrt(input);
        }


        /// <summary>find the element of <paramref name="source"/> which has the highest value of <paramref name="withRespectTo"/></summary>
        /// <typeparam name="TSrc"></typeparam>
        /// <param name="source">The source</param>
        /// <param name="withRespectTo">The function for which we want to find the element of <paramref name="@this"/> which results in the highest value.</param>
        /// <returns>The element of <paramref name="@this"/> for which <paramref name="withRespectTo"/> is maximised. If all elements result in the same value, returns the first element of <paramref name="source"/></returns>
        public static TSrc ArgMax<TSrc>(this IEnumerable<TSrc> source, Func<TSrc, double> withRespectTo)
        {
            return source.Aggregate(source.First(), (argmax, el) => withRespectTo(el) > withRespectTo(argmax) ? el : argmax);
        }

        /// <summary>find the index of the element of <paramref name="source"/> which has the highest value of <paramref name="withRespectTo"/></summary>
        /// <typeparam name="TSrc"></typeparam>
        /// <param name="source">The source</param>
        /// <param name="withRespectTo">The function for which we want to find the element of <paramref name="this"/> which results in the highest value.</param>
        /// <returns>The index of the element of <paramref name="this"/> for which <paramref name="withRespectTo"/> is maximised. If all elements result in the same value, returns 0.</returns>
        public static int ArgMaxIndex<TSrc>(this IEnumerable<TSrc> @this, Func<TSrc, double> withRespectTo)
        {
            return
                @this.Select((el, i) => new KeyValuePair<TSrc, int>(el, i))
                     .Aggregate(
                                (argmaxIndex, el) => withRespectTo(el.Key) > withRespectTo(argmaxIndex.Key) ? el : argmaxIndex)
                     .Value;
        }
    }
}