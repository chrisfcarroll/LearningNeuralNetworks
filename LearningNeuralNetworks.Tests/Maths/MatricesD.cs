using System;
using System.Linq;
using LearningNeuralNetworks.Frameworks;
using LearningNeuralNetworks.Maths;
using NUnit.Framework;
using TestBase.Shoulds;

namespace LearningNeuralNetworks.Tests.Maths
{
    [TestFixture]
    public class MatricesD
    {
        [TestCase(2, 3)]
        public void MatrixEqualityOperator__IsCorrect(int rows, int columns)
        {
            var left  = new MatrixD(new double[,] { {1,2}, {3,4} });
            var right = new MatrixD(new double[,] { { 1, 2 }, { 3, 4 } });
            (left == right).ShouldBeTrue();
            (left != right).ShouldBeFalse();
        }

        [TestCase(1, 2, 3, 4, 5, 6, 7, 8, 6,8,10,12)]
        public void TwoSquareMatricesAdd(params double[] cells)
        {
            int size = (int) Math.Sqrt(cells.Length/3);
            ( size*size*3 == cells.Length).ElseThrow("this test needs parameters to form 3 square matrices, e.g. 3 or 12 or 27 ... parameters");
            //
            var left= new MatrixD(size,size, cells.Take(size*size).ToArray());
            var right= new MatrixD(size,size, cells.Skip(size*size).Take(size*size).ToArray());
            var expected = new MatrixD(size, size, cells.Skip(size*size*2).Take(size*size).ToArray());
            //
            (left + right).ShouldEqualByValue(expected);
        }

        [TestCase(1, 0, 0, 1, 5, 6, 7, 8, 5, 6, 7, 8)]
        public void TwoSquareMatricesMultiply(params double[] cells)
        {
            int size = (int)Math.Sqrt(cells.Length / 3);
            (size * size * 3 == cells.Length).ElseThrow("this test needs parameters to form 3 square matrices, e.g. 3 or 12 or 27 ... parameters");
            //
            var left = new MatrixD(size, size, cells.Take(size * size).ToArray());
            var right = new MatrixD(size, size, cells.Skip(size * size).Take(size * size).ToArray());
            var expected = new MatrixD(size, size, cells.Skip(size * size * 2).Take(size * size).ToArray());
            //
            (left * right).ShouldEqualByValue(expected);
        }

        [TestCase(1, 2, 3, 4, 5, 6, 1, 2, 3, 14, 32)]
        public void MatrixDotProduct23x31__IsCorrect(params double[] cells)
        {
            var left = new MatrixD(2,3, cells.Take(6).ToArray());
            var right = new MatrixD(3,1, cells.Skip(6).Take(3).ToArray());
            var expected = new MatrixD(2,1, cells.Skip(9).Take(2).ToArray());
            //
            (left * right).ShouldEqualByValue(expected);
        }

    }
}
