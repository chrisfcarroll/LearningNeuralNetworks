namespace LearningNeuralNetworks.Maths
{
    public static class MatrixD_ArrayExtensions
    {
        public static MatrixD AsMatrixD(this double[][] array) {  return new MatrixD(array);}
    }
}