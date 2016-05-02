using System;
using System.Collections.Generic;
using System.Linq;

namespace LearningNeuralNetworks
{
    public static class MnistSigmoidLearner1NetBuilderTrainer
    {
        public static InterpretedNet<byte> Build(int hiddenLayerSize)
        {
            return new InterpretedNet<byte>(
                        new NeuralNet3LayerSigmoid(784, hiddenLayerSize, 10),
                        e=>(byte) e.ArgMaxIndex(n=>n),
                        b => Enumerable.Range(0,10).Select(i => i==b? 1 :0).Select(i => (ZeroToOne)i),
                        Comparer<byte>.Default
                        ) ;
        }
    }

    public static class EnumerableExtensions
    {
        /// <summary>find the element of <paramref name="source"/> which has the highest value of <paramref name="withRespectTo"/></summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source">The source</param>
        /// <param name="withRespectTo">The function for which we want to find the element of <paramref name="@this"/> which results in the highest value.</param>
        /// <returns>The element of <paramref name="@this"/> for which <paramref name="withRespectTo"/> is maximised. If all elements result in the same value, returns the first element of <paramref name="source"/></returns>
        public static T ArgMax<T>(this IEnumerable<T> source, Func<T,double> withRespectTo)
        {
            return source.Aggregate(source.First(), (argmax, el) => withRespectTo(el) > withRespectTo(argmax) ? el : argmax);
        }

        /// <summary>find the index of the element of <paramref name="source"/> which has the highest value of <paramref name="withRespectTo"/></summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source">The source</param>
        /// <param name="withRespectTo">The function for which we want to find the element of <paramref name="@this"/> which results in the highest value.</param>
        /// <returns>The index of the element of <paramref name="@this"/> for which <paramref name="withRespectTo"/> is maximised. If all elements result in the same value, returns 0.</returns>
        public static int ArgMaxIndex<T>(this IEnumerable<T> @this, Func<T, double> withRespectTo)
        {
            return
                @this.Select((el, i) => new KeyValuePair<T, int>(el, i))
                    .Aggregate(
                        (argmaxIndex, el) => withRespectTo(el.Key) > withRespectTo(argmaxIndex.Key) ? el : argmaxIndex)
                    .Value;
        }
    }

}
