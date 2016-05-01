namespace LearningNeuralNetworks
{
    public class MatrixL
    {
        public long this[int rowIndex, int columnIndex] { get { return data[rowIndex, columnIndex]; } set { data[rowIndex, columnIndex] = value; } }
        public int ColumnCount { get; private set; }
        public int RowCount { get; private set; }

        public MatrixL(int rowCount, int columnCount)
        {
            RowCount = rowCount;
            ColumnCount = columnCount;
            data = new long[rowCount, columnCount];
        }

        readonly long[,] data;
    }

    public class MatrixF
    {
        public double this[int rowIndex, int columnIndex] { get { return data[rowIndex, columnIndex]; } set { data[rowIndex, columnIndex] = value; } }
        public int ColumnCount { get; private set; }
        public int RowCount { get; private set; }

        public MatrixF(int rowCount, int columnCount)
        {
            RowCount = rowCount;
            ColumnCount = columnCount;
            data = new double[rowCount, columnCount];
        }

        readonly double[,] data;
    }
}
