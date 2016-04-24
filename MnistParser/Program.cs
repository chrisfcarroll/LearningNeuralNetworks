using System.Linq;

namespace MnistParser
{
    public class Program
    {
        public static void Main(params string[] args)
        {
            var reader = ReaderFromDirectoryAndFileNamesElseConfig(
                            args?.Length > 0 ? args[0] : Properties.Settings.Default.MnistDataDirectory, 
                            args.Skip(1).ToArray()
                        );

            reader.EnsureLoaded();
        }

        public static MnistFilesReader ReaderFromDirectoryAndFileNamesElseConfig(string mnistSource, string[] args=null)
        {
            MnistSource = mnistSource;
            //
            if (args!=null && args.Length >= 4)
            {
                TrainImagesIdx3Ubyte = args[1];
                TrainLabelsIdx1Ubyte = args[2];
                T10KImagesIdx3Ubyte = args[3];
                T10KLabelsIdx1Ubyte = args[4];
            }
            else
            {
                TrainImagesIdx3Ubyte = Properties.Settings.Default.trainImagesFileName;
                TrainLabelsIdx1Ubyte = Properties.Settings.Default.trainLabelsFileName;
                T10KImagesIdx3Ubyte = Properties.Settings.Default.testImagesFileName;
                T10KLabelsIdx1Ubyte = Properties.Settings.Default.testLabelsFileName;
            }

            return new MnistFilesReader(mnistSource, TrainImagesIdx3Ubyte, TrainLabelsIdx1Ubyte, T10KImagesIdx3Ubyte, T10KLabelsIdx1Ubyte);
        }

        public static string MnistSource;
        public static string TrainImagesIdx3Ubyte;
        public static string TrainLabelsIdx1Ubyte;
        public static string T10KImagesIdx3Ubyte;
        public static string T10KLabelsIdx1Ubyte;

    }
}
