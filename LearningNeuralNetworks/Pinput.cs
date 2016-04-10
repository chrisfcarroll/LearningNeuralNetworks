namespace LearningNeuralNetworks
{
    public struct Pinput
    {
        public int weight;
        public Perceptron Source;
        public Pinput(Perceptron source, int weight) : this()
        {
            this.weight = weight;
            this.Source = source;
        }
    }
}