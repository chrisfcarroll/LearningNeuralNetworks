using System;
using System.Linq;
using System.Runtime.InteropServices;
using LearningNeuralNetworks.Frameworks;

namespace LearningNeuralNetworks.Maths
{
    /// <summary>A Matrix of System.Double</summary>
    public partial class MatrixD 
    {
        public static implicit operator double[][] (MatrixD matrix) { return matrix.ByRows(); }
        public static implicit operator MatrixD(double[][] double2dArray) { return new MatrixD(double2dArray); }

        public static MatrixD NewRandom(int rowCount, int columnCount, [Optional] Random random, double scale=1d, double centredOn = 0d)
        {
            var rnd = random?? new Random();
            var result = new double[rowCount][];
            for (int i = 0; i < rowCount; i++)
            {
                result[i] = Enumerable.Range(0, columnCount).Select(j => rnd.NextDouble() * scale + centredOn - 0.5).ToArray();
            }
            return new MatrixD(result);
        }

        public MatrixD(int rowCount, int columnCount)
        {
            RowCount = rowCount;
            ColumnCount = columnCount;
            data = Enumerable.Range(0,rowCount).Select(i => new double[columnCount]).ToArray();
        }

        /// <summary>
        /// Returns a Matrix with all elements initialised to <paramref name="initialValueOfAllElements"/>
        /// </summary>
        /// <param name="rowCount"></param>
        /// <param name="columnCount"></param>
        /// <param name="initialValueOfAllElements"></param>
        public MatrixD(int rowCount, int columnCount, double initialValueOfAllElements)
        {
            var row = Enumerable.Repeat(initialValueOfAllElements, columnCount).ToArray();
            RowCount = rowCount;
            ColumnCount = columnCount;
            data = new double[rowCount][];
            for (int i = 0; i < rowCount; i++)
            {
                data[i] = new double[columnCount];
                Array.Copy(row,data[i],columnCount);
            }
        }

        /// <summary>
        /// turns e.g. 
        /// (2,3, [1,2,3,4,5,6]) into a 2*3 matrixk [[1,2,3], [4,5,6]]
        /// (3,2, [1,2,3,4,5,6]) into a 3*2 matrixk [[1,2],[3,4], [5,6]]
        /// </summary>
        /// <param name="rowCount"></param>
        /// <param name="columnCount"></param>
        /// <param name="flattenedData">the data which is read as if it were the rows of the matrix concatenated into a 1d array </param>
        public MatrixD(int rowCount, int columnCount, double[] flattenedData): this(rowCount,columnCount)
        {
            (rowCount*columnCount== flattenedData.Length).ElseThrow( new ArgumentOutOfRangeException(nameof(flattenedData),$"the given array is size {flattenedData.Length} but must be of size {rowCount}*{columnCount}"));
            //
            for(int i = 0; i<RowCount; i++)
            for(int j = 0; j<ColumnCount; j++)
            {
                data[i][j] = flattenedData[i*ColumnCount + j];
            }
        }


        public MatrixD(double[][] dataSourceByReference)
        {
            data = dataSourceByReference;
            RowCount = dataSourceByReference.Length;
            try
            {
                ColumnCount = dataSourceByReference[0].Length;
                (ColumnCount== dataSourceByReference[RowCount-1].Length).ElseThrow(new ArgumentException("The given array must be rectangular, but this has jagged array lengths", nameof(dataSourceByReference)));
            }
            catch (Exception e) { throw new ArgumentException("The given array must be rectangular, but this isn't", nameof(dataSourceByReference), e); }
        }

        public MatrixD(double[,] dataSourceByReference)
        {
            RowCount = dataSourceByReference.GetLength(0);
            ColumnCount = dataSourceByReference.GetLength(1);
            data = Copy2DToJagged(dataSourceByReference);
        }

        public MatrixD Copy() { return new MatrixD(data.Select(r=> r.ToArray()).ToArray()); }

        unsafe static double[][] Copy2DToJagged(double[,] rect)
        {
            // ReSharper disable SuggestVarOrType_BuiltInTypes
            // ReSharper disable SuggestVarOrType_Elsewhere
            int
                row1 = rect.GetLowerBound(0),
                rowN = rect.GetUpperBound(0),
                col1 = rect.GetLowerBound(1),
                colN = rect.GetUpperBound(1);
            int height = rowN - row1 + 1;
            int width = colN - col1 + 1;

            double[][] jagged = new double[height][];
            int k = 0;
            for (int i = row1; i < row1 + height; i++)
            {
                double[] temp = new double[width];

                fixed (double* dest = temp, src = &rect[i, col1])
                {
                    MoveMemory(dest, src, rowN * sizeof(double));
                }
                jagged[k++] = temp;
            }
            return jagged;

            // ReSharper restore SuggestVarOrType_Elsewhere
            // ReSharper restore SuggestVarOrType_BuiltInTypes
        }

        [DllImport("kernel32.dll", EntryPoint = "RtlMoveMemory")]
        unsafe internal static extern void MoveMemory(void* dest, void* src, int length);
    }
}
