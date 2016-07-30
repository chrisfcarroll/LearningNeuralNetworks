using System;
using System.Text;

namespace LearningNeuralNetworks.Maths
{
    /// <summary>A Matrix of System.Double</summary>
    public partial class MatrixD
    {
        public double[][] ByRows()
        {
            return Copy().data;
        }

        public override string ToString()
        {
            return ToString("G");
        }

        public string ToString(string format)
        {
            var sb = new StringBuilder("[");
            for (int i = 0; i < RowCount; i++)
            {
                sb.Append('[');
                for (int j = 0; j < ColumnCount; j++)
                {
                    sb.Append(data[i][j].ToString(format)).Append(',').Append(' ');
                }
                sb.Append(']').AppendLine();
            }
            return sb.Append(']').AppendLine().ToString();
        }

        public string ToString(int maxRowsToShow, int maxColumnsToShow, string format = "G")
        {
            maxRowsToShow = Math.Min(RowCount, maxRowsToShow);
            maxColumnsToShow = Math.Min(ColumnCount, maxColumnsToShow);
            var halfRows = Math.Max(1, (1 + maxRowsToShow) / 2);
            var halfCols = Math.Max(1, (1 + maxColumnsToShow) / 2);
            var isElidingCols = ColumnCount > 2 * maxColumnsToShow;
            var isElidingRows = RowCount > 2 * maxRowsToShow;
            var topRows = new { from = 0, to = halfRows };
            var bottomRows = new { from = RowCount - halfRows, to = RowCount };
            //
            var sb = new StringBuilder("[");
            foreach (var range in new[] { topRows, bottomRows })
            {
                for (int i = range.from; i < range.to; i++)
                {
                    sb.Append('[');
                    for (int j = 0; j < halfCols; j++)
                    {
                        sb.Append(data[i][j].ToString(format)).Append(',').Append(' ');
                    }
                    if (isElidingCols) { sb.Append('…'); }
                    for (int j = ColumnCount - halfCols; j < ColumnCount; j++)
                    {
                        sb.Append(data[i][j].ToString(format)).Append(',').Append(' ');
                    }
                    sb.Append(']').AppendLine();
                }
                if (isElidingRows && range == topRows) for (int k = 0; k < maxColumnsToShow; k++) { sb.Append(" … "); }
            }
            return sb.Append(']').AppendLine().ToString();
        }
    }
}
