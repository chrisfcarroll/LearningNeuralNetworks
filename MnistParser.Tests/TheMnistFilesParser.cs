using System.Linq;
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
                Program.ReaderFromDirectoryAndFileNamesElseConfig(mnistRealDataDirectory,maxImagesToRead:2);
                //
                Program.MnistSource.ShouldBe(mnistRealDataDirectory);
                Program.TrainImagesIdx3Ubyte.ShouldNotBeEmpty();
                Program.TrainLabelsIdx1Ubyte.ShouldNotBeEmpty();
                Program.T10KImagesIdx3Ubyte.ShouldNotBeEmpty();
                Program.T10KLabelsIdx1Ubyte.ShouldNotBeEmpty();
            }

            [Test]
            public void ParsesAndExposesTrainingLabelFile()
            {
                SetUpLoadedReader();
                //
                reader.TrainingLabels.ShouldNotBeEmpty();
            }

            [Test]
            public void ParsesAndExposesTrainingImageFile()
            {
                SetUpLoadedReader();
                //
                reader.TrainingImages.ShouldNotBeEmpty().First().Data.ShouldNotBeEmpty();
            }

            [Test]
            public void ParsesAndExposesTestLabelFile()
            {
                SetUpLoadedReader();
                //
                reader.TestLabels.ShouldNotBeEmpty();
            }

            [Test]
            public void ParsesAndExposesTestImageFile()
            {
                SetUpLoadedReader();
                //
                reader.TestImages.ShouldNotBeEmpty().First().Data.ShouldNotBeEmpty();
            }


            void SetUpLoadedReader()
            {
                reader= (reader !=null && reader.IsLoaded) 
                            ? reader 
                            : new MnistFilesReader(mnistRealDataDirectory,2).EnsureLoaded();
            }
        }
    }
}
