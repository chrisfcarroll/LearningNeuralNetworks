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
        public int MaximumImagesToRead { get; private set; }
        public bool IsLoaded { get; private set; }
        public Image[] TrainingImages { get; private set; }
        public byte[] TrainingLabels { get; private set; }
        public Image[] TestImages { get; private set; }
        public byte[] TestLabels { get; private set; }

        readonly string dataDirectory;
        readonly string trainingImagesFileName;
        readonly string trainingLabelsFileName;
        readonly string testImagesFileName;
        readonly string testLabelsFileName;
        /// <param name="mnistDataDirectory">The directory in which to find the four Mnist files</param>
        public MnistFilesReader(string mnistDataDirectory, string trainingImagesFileName, string trainingLabelsFileName, string testImagesFileName, string testLabelsFileName, int maximumImagesToRead=int.MaxValue)
        {
            MaximumImagesToRead = maximumImagesToRead;
            this.dataDirectory = mnistDataDirectory;
            this.trainingImagesFileName = trainingImagesFileName;
            this.trainingLabelsFileName = trainingLabelsFileName;
            this.testImagesFileName = testImagesFileName;
            this.testLabelsFileName = testLabelsFileName;
            MaximumImagesToRead = maximumImagesToRead;
        }
        /// <param name="mnistDataDirectory">The directory in which to find the four Mnist files</param>
        public MnistFilesReader(string mnistDataDirectory = null) 
            : this(
                  mnistDataDirectory ?? Properties.Settings.Default.MnistDataDirectory,
                  Properties.Settings.Default.trainImagesFileName,
                  Properties.Settings.Default.trainLabelsFileName,
                  Properties.Settings.Default.testImagesFileName,
                  Properties.Settings.Default.testLabelsFileName
                  ){}

        public MnistFilesReader(string mnistRealDataDirectory, int maximumImagesToRead) 
            : this(
                  mnistRealDataDirectory ?? Properties.Settings.Default.MnistDataDirectory,
                  Properties.Settings.Default.trainImagesFileName,
                  Properties.Settings.Default.trainLabelsFileName,
                  Properties.Settings.Default.testImagesFileName,
                  Properties.Settings.Default.testLabelsFileName,
                  maximumImagesToRead: maximumImagesToRead) {}

        internal MnistFilesReader EnsureLoaded()
        {
            if (!IsLoaded)
            {
                Task.WaitAll(
                    Task.Run(async () => { TrainingLabels = await ParseLabelsFileAsync(trainingLabelsFileName); }),
                    Task.Run(async () => { TrainingImages = await ParseImagesFileAsync(trainingImagesFileName); }),
                    Task.Run(async () => { TestLabels = await ParseLabelsFileAsync(testLabelsFileName); }),
                    Task.Run(async () => { TestImages = await ParseImagesFileAsync(testImagesFileName); })
                    );
                IsLoaded = true;
            }
            return this;
        }


        async Task<Image[]> ParseImagesFileAsync(string imagesFileName)
        {
            byte[] buffer1 = new byte[4];
            using (var fileStream = new FileStream(Path.Combine(dataDirectory, imagesFileName), FileMode.Open, FileAccess.Read, FileShare.Read, 4096, useAsync: true))
            using (var stream = new BufferedStream(fileStream))
            {
                stream.Read(buffer1, 0, 4); var magicNumber = buffer1[2] * 0x100 + buffer1[3];
                stream.Read(buffer1, 0, 4); var numOfImages = buffer1[2] * 0x100 + buffer1[3];
                stream.Read(buffer1, 0, 4); var ySize = buffer1[2] * 0x100 + buffer1[3];
                stream.Read(buffer1, 0, 4); var xSize = buffer1[2] * 0x100 + buffer1[3];
                //
                EnsureImagesFileHeaderAndMagicNumber(magicNumber, numOfImages, xSize, ySize, imagesFileName);
                var trainingImages = new Image[numOfImages];
                var i = 0;
                var imageBuffer = new byte[Image.ByteSize];
                while (await stream.ReadAsync(imageBuffer, 0, Image.ByteSize) > 0 && i < MaximumImagesToRead)
                {
                    Debug.Assert(i < numOfImages, "Expected " + numOfImages + " images but about to read past that.");
                    trainingImages[i] = new Image(imageBuffer);
                    i++;
                }
                Debug.Assert( i == Math.Min(numOfImages, MaximumImagesToRead), "Expected " + Math.Min(numOfImages, MaximumImagesToRead) + " images but got " + i);
                return trainingImages;
            }
        }

        async Task<byte[]> ParseLabelsFileAsync(string labelsFileName)
        {
            byte[] buffer = new byte[4];
            using (var fileStream = new FileStream(Path.Combine(dataDirectory, labelsFileName), FileMode.Open, FileAccess.Read, FileShare.Read, 4096, useAsync: true))
            using (var stream = new BufferedStream(fileStream))
            {
                stream.Read(buffer, 0, 4); var magicNumber = buffer[2] * 0x100 + buffer[3];
                stream.Read(buffer, 0, 4); var numOfLabels = buffer[2] * 0x100 + buffer[3];
                //
                EnsureLabelsFileHeaderAndMagicNumber(magicNumber, numOfLabels, labelsFileName);
                var trainingLabels = new byte[numOfLabels];
                var i = 0;
                while (await stream.ReadAsync(buffer, 0, 1) > 0 && i < MaximumImagesToRead)
                {
                    trainingLabels[i] = buffer[0];
                    i++;
                }
                Debug.Assert(i == Math.Min(numOfLabels, MaximumImagesToRead), "Expected " + Math.Min(numOfLabels, MaximumImagesToRead) + " labels but got " + i);
                return trainingLabels;
            }
        }


        void EnsureLabelsFileHeaderAndMagicNumber(int magicNumber, int numOfLabels, string fileName)
        {
            Debug.Assert(magicNumber == 0x00000801, fileName + " didn't start with the magic number 0x00000801 so it can't be an Mnist training label file");
            Debug.Assert(numOfLabels > 0,  fileName + " claimed to label " + numOfLabels + " images, but I expected more than zero");
        }

        void EnsureImagesFileHeaderAndMagicNumber(int magicNumber, int numOfImages, int xSize, int ySize, string fileName)
        {
            Debug.Assert(magicNumber == 0x00000803, fileName + " didn't start with the magic number 0x00000803 so it can't be an Mnist training images file");
            Debug.Assert(numOfImages > 0, fileName + " claimed to be " + numOfImages + " images, but I expected more than zero");
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