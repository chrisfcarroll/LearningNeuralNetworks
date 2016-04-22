using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

namespace MnistParser
{
    /// <summary>
    /// Read the four <see href="http://yann.lecun.com/exdb/mnist/">Mnist data files</see> from a given directory
    /// </summary>
    public class MnistFilesReader
    {
        public byte[] TrainingLabels { get; private set; }

        readonly string dataDirectory;
        readonly string trainingDataFileName;
        readonly string trainingLabelsFileName;
        readonly string testDataFileName;
        readonly string testLabelsFileName;
        /// <param name="mnistDataDirectory">The directory in which to find the four Mnist files</param>
        public MnistFilesReader(string mnistDataDirectory, string trainingDataFileName, string trainingLabelsFileName, string testDataFileName, string testLabelsFileName)
        {
            this.dataDirectory = mnistDataDirectory;
            this.trainingDataFileName = trainingDataFileName;
            this.trainingLabelsFileName = trainingLabelsFileName;
            this.testDataFileName = testDataFileName;
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

        internal MnistFilesReader Load()
        {
            Task.WaitAll( 
                Task.Run(async () => { TrainingLabels = await ParseTrainingLabelsFileAsync(); }) 
                );
            return this;
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
                EnsureTrainingLabelsMagicNumber(magicNumber);
                EnsureSomeTrainingLabels(numOfLabels);
                //
                var trainingLabels = new byte[numOfLabels];
                var i = 0;
                while (await stream.ReadAsync(buffer, 0, 1) > 0)
                {
                    trainingLabels[i] = buffer[0];
                    i++;
                }
                return trainingLabels;
            }
        }

        void EnsureSomeTrainingLabels(int numOfLabels)
        {
            Debug.Assert(numOfLabels > 0,  trainingLabelsFileName + " claimed to label " + numOfLabels + " images, but I expected more than zero");
        }

        void EnsureTrainingLabelsMagicNumber(int magicNumber)
        {
            Debug.Assert(magicNumber == 0x00000801, trainingLabelsFileName + " didn't start with the magic number 0x00000801 so it can't be an Mnist training label file");
        }
    }
}