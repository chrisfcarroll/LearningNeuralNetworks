using NUnit.Framework;
using TestBase.Shoulds;

namespace MnistParser.Tests
{
    public class TheMnistFilesParser
    {
        [TestFixture]
        public class GivenADataDirectoryWithFourFiles
        {
            static string mnistRealDataDirectory = @"..\..\..\MnistParser\MnistData";
            static MnistFilesReader reader;

            [Test]
            public void UsesFourDefaultFileNames__GivenADataDirectory()
            {
                Program.ReaderFromDirectoryAndFileNamesElseConfig(mnistRealDataDirectory);
                //
                Program.MnistSource.ShouldBe(mnistRealDataDirectory);
                Program.TrainImagesIdx3Ubyte.ShouldNotBeEmpty();
                Program.TrainLabelsIdx1Ubyte.ShouldNotBeEmpty();
                Program.T10KImagesIdx3Ubyte.ShouldNotBeEmpty();
                Program.T10KLabelsIdx1Ubyte.ShouldNotBeEmpty();
            }

            [Test]
            public void ParsesTrainingLabelFile()
            {
                SetUpLoadedReader();
                //
                reader.TrainingLabels.ShouldNotBeEmpty();
            }

            [Test]
            public void ParsesTrainingImageFile()
            {
                SetUpLoadedReader();
                //
                reader.TrainingImages.ShouldNotBeEmpty();
            }

            void SetUpLoadedReader()
            {
                reader= (reader !=null && reader.IsLoaded) 
                            ? reader 
                            : new MnistFilesReader(mnistRealDataDirectory).EnsureLoaded();
            }
        }
    }
}
