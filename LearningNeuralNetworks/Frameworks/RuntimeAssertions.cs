using System;

namespace LearningNeuralNetworks.Frameworks
{
    /// <summary>
    /// Throw runtime exceptions
    /// </summary>
    public static class RuntimeAssertions
    {
        public static bool ElseThrow(this bool assertion, Exception exception) { if (assertion) return true; else throw exception; }

        public static bool ElseThrow(this bool assertion, string message){ return ElseThrow(assertion, new AssertionFailedException(message)); }

        public class AssertionFailedException : Exception
        {
            public AssertionFailedException(string message) : base(message)
            {
            }
        }
    }
}
