using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;

namespace MnistParser
{
    public class Image
    {
        public const int ByteSizeX = 28;
        public const int ByteSizeY = 28;
        public const int ByteSize = ByteSizeX * ByteSizeY;


        public byte[]   As1DBytes => Enumerated().ToArray();
        public double[] As1Ddoubles => Enumerated().Select(x=> (double)x).ToArray();
        public byte[,] Data { get; }


        public Image(byte[,] imageData28x28pixels)
        {
            Contract.Requires(imageData28x28pixels.GetLength(0)==ByteSizeX );
            Contract.Requires(imageData28x28pixels.GetLength(1) == ByteSizeY);
            //
            Data = imageData28x28pixels;
        }

        public Image(byte[] imageData28x28pixels)
        {
            Contract.Requires(imageData28x28pixels.Length == ByteSize);
            Data= new byte[ByteSizeX,ByteSizeY];
            for (int i = 0; i < ByteSizeX; i++)
                for (int j = 0; j < ByteSizeX; j++)
                {
                    Data[i, j] = imageData28x28pixels[i*ByteSizeX + j];
                }
        }

        IEnumerable<byte> Enumerated()
        {
            for (int i = 0; i < ByteSizeX; i++)
                for (int j = 0; j < ByteSizeY; j++)
                {
                    yield return Data[i, j];
                }
        }
    }
}