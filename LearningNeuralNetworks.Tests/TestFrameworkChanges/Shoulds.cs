namespace LearningNeuralNetworks.Tests.TestFrameworkChanges
{
    public static class Shoulds
    {
        /// <summary>This test can probably only fail if compilation fails. That is the intent, at least</summary>
        /// <returns><param name="input"></param></returns>
        public static T ShouldCompile<T>(this T input) { return input; }
    }
}
