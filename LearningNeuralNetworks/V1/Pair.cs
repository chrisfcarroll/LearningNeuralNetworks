using System.Collections.Generic;

namespace LearningNeuralNetworks.V1
{
    public struct Pair<TData,TLabel>
    {
        public TData Data  { get; }
        public TLabel Label { get; }

        /// <summary>Initializes a new instance of the <see cref="Pair"/> structure with the specified data and label.</summary>
        public Pair(TData data, TLabel label) { Data = data; Label = label; }

        public override string ToString() { return "<" + Data + "," + Label + ">"; }



        public static implicit operator KeyValuePair<TData, TLabel>(Pair<TData, TLabel> pair)
        {
            return new KeyValuePair<TData, TLabel>(pair.Data, pair.Label);
        }
        public static implicit operator Pair<TData, TLabel>(KeyValuePair<TData, TLabel> kvpair)
        {
            return new Pair<TData, TLabel>(kvpair.Key, kvpair.Value);
        }
    }

    public static class Pair
    {
        public static Pair<TD, TL> New<TD, TL>(TD data, TL label) { return new Pair<TD, TL>(data, label); }
    }
}