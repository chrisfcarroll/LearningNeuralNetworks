using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace LearningNeuralNetworks.Frameworks
{
    public static class EnumerableExtensions
    {
        /// <summary> Synonym for <see cref="Enumerable.Reverse{TSource}"/> because List{T}.Reverse() returns void. </summary>
        /// <returns><paramref name="source"/> in reverse order</returns>
        public static IEnumerable<T> OrderByReverse<T>(this IEnumerable<T> source) => source.Reverse();

        /// <summary>
        /// Sorts the IEnumerable randomly
        /// </summary>
        /// <typeparam name="TData"></typeparam>
        /// <typeparam name="TLabel"></typeparam>
        /// <param name="source"></param>
        /// <param name="random">The pseudo-random number generator to use.</param>
        /// <returns></returns>
        public static IEnumerable<T> OrderRandomly<T>(this IEnumerable<T> source, [Optional] Random random)
        {
            if(source==null)throw new ArgumentNullException(nameof(source),"Passed a null source to OrderRandomly");
            //
            var populationSize = source.Count();
            random = random??new Random();
            var batch = Enumerable
                            .Range(0, populationSize)
                            .OrderBy(i => random.Next())
                            .Select(i => source.ElementAt(i));
            return batch;
        }
    }
}
