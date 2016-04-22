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

            [Test]
            public void UsesFourDefaultFileNames__GivenADataDirectory()
            {
                Program.Main(mnistRealDataDirectory);
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
                var reader = new MnistFilesReader(mnistRealDataDirectory).Load();
                //
                reader.TrainingLabels.ShouldNotBeEmpty();
            }
        }
    }
}
