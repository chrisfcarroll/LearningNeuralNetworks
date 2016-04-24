using System;
using System.Diagnostics;
using System.Diagnostics.Contracts;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;

namespace MnistParser
{
    /// <summary>
    /// Read the four <see href="http://yann.lecun.com/exdb/mnist/">Mnist data files</see> from a given directory
    /// </summary>
    public class MnistFilesReader
    {
        public bool IsLoaded { get; private set; }
        public Image[] TrainingImages { get; private set; }
        public byte[] TrainingLabels { get; private set; }

        readonly string dataDirectory;
        readonly string trainingImagesFileName;
        readonly string trainingLabelsFileName;
        readonly string testImagesFileName;
        readonly string testLabelsFileName;
        /// <param name="mnistDataDirectory">The directory in which to find the four Mnist files</param>
        public MnistFilesReader(string mnistDataDirectory, string trainingImagesFileName, string trainingLabelsFileName, string testImagesFileName, string testLabelsFileName)
        {
            this.dataDirectory = mnistDataDirectory;
            this.trainingImagesFileName = trainingImagesFileName;
            this.trainingLabelsFileName = trainingLabelsFileName;
            this.testImagesFileName = testImagesFileName;
            this.testLabelsFileName = testLabelsFileName;
        }
        /// <param name="mnistDataDirectory">The directory in which to find the four Mnist files</param>
        public MnistFilesReader(string mnistDataDirectory = null) 
            : this(
                  mnistDataDirectory ?? Properties.Settings.Default.MnistDataDirectory,
                  Properties.Settings.Default.trainImagesFileName,
                  Properties.Settings.Default.trainLabelsFileName,
                  Properties.Settings.Default.testImagesFileName,
                  Properties.Settings.Default.testLabelsFileName
                  ){ }

        internal MnistFilesReader EnsureLoaded()
        {
            if (IsLoaded){return this;}
            //
            Task.WaitAll(
                Task.Run(async () => { TrainingLabels = await ParseTrainingLabelsFileAsync(); }),
                Task.Run(async () => { TrainingImages = await ParseTrainingImagesFileAsync(); })
                );
            IsLoaded = true;
            //TrainingLabels = ParseTrainingLabelsFileAsync().Result;
            //TrainingImages = ParseTrainingImagesFileAsync().Result;
            return this;
        }


        async Task<Image[]> ParseTrainingImagesFileAsync()
        {
            byte[] buffer1 = new byte[4];
            using (var fileStream = new FileStream(Path.Combine(dataDirectory, trainingImagesFileName), FileMode.Open, FileAccess.Read, FileShare.Read, 4096, useAsync: true))
            using (var stream = new BufferedStream(fileStream))
            {
                stream.Read(buffer1, 0, 4); var magicNumber = buffer1[2] * 0x100 + buffer1[3];
                stream.Read(buffer1, 0, 4); var numOfImages = buffer1[2] * 0x100 + buffer1[3];
                stream.Read(buffer1, 0, 4); var ySize = buffer1[2] * 0x100 + buffer1[3];
                stream.Read(buffer1, 0, 4); var xSize = buffer1[2] * 0x100 + buffer1[3];
                //
                EnsureTrainingImagesFileHeaderAsExpected(magicNumber, numOfImages, xSize, ySize);
                //
                var trainingImages = new Image[numOfImages];
                var i = 0;
                var imageBuffer = new byte[Image.ByteSize];
                while (await stream.ReadAsync(imageBuffer, 0, Image.ByteSize) > 0)
                {
                    Debug.Assert(i < numOfImages, "Expected " + numOfImages + " images but about to read past that.");
                    trainingImages[i] = new Image(imageBuffer);
                    i++;
                }
                Debug.Assert( i == numOfImages, "Expected " + numOfImages + " images but got " + i);
                return trainingImages;
            }
        }

        async Task<byte[]> ParseTrainingLabelsFileAsync()
        {
            byte[] buffer = new byte[4];
            using (var fileStream = new FileStream(Path.Combine(dataDirectory, trainingLabelsFileName), FileMode.Open, FileAccess.Read, FileShare.Read, 4096, useAsync: true))
            using (var stream = new BufferedStream(fileStream))
            {
                stream.Read(buffer, 0, 4); var magicNumber = buffer[2] * 0x100 + buffer[3];
                stream.Read(buffer, 0, 4); var numOfLabels = buffer[2] * 0x100 + buffer[3];
                //
                EnsureTrainingLabelsHeaderAsExpected(magicNumber, numOfLabels);
                //
                var trainingLabels = new byte[numOfLabels];
                var i = 0;
                while (await stream.ReadAsync(buffer, 0, 1) > 0)
                {
                    trainingLabels[i] = buffer[0];
                    i++;
                }
                Debug.Assert(i == numOfLabels, "Expected " + numOfLabels + " labels but got " + i);
                return trainingLabels;
            }
        }


        void EnsureTrainingLabelsHeaderAsExpected(int magicNumber, int numOfLabels)
        {
            Debug.Assert(magicNumber == 0x00000801, trainingLabelsFileName + " didn't start with the magic number 0x00000801 so it can't be an Mnist training label file");
            Debug.Assert(numOfLabels > 0,  trainingLabelsFileName + " claimed to label " + numOfLabels + " images, but I expected more than zero");
        }

        void EnsureTrainingImagesFileHeaderAsExpected(int magicNumber, int numOfImages, int xSize, int ySize)
        {
            Debug.Assert(magicNumber == 0x00000803, trainingImagesFileName + " didn't start with the magic number 0x00000803 so it can't be an Mnist training images file");
            Debug.Assert(numOfImages > 0, trainingImagesFileName + " claimed to be " + numOfImages + " images, but I expected more than zero");
            Debug.Assert(xSize == 28 && ySize == 28, "Expected image sizes 28x28, but got " + xSize + "x" + ySize);
        }
    }

    public class Image
    {
        public const int ByteSizeX = 28;
        public const int ByteSizeY = 28;
        public const int ByteSize = ByteSizeX * ByteSizeY;

        public byte[,] Data { get; private set; }

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

    }
}