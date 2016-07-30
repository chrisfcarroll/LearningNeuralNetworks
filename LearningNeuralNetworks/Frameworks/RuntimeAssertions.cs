using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningNeuralNetworks.Frameworks
{
    /// <summary>
    /// Throw runtime exceptions
    /// </summary>
    public static class RuntimeAssertions
    {
        public static void ElseThrow(this bool assertion, string message)
        {
            ElseThrow(assertion, new AssertionFailedException(message));
        }

        public static void ElseThrow(this bool assertion, Exception exception)
        {
            if (!assertion) throw exception;
        }

        public class AssertionFailedException : Exception
        {
            public AssertionFailedException(string message) : base(message)
            {
            }
        }
    }
}
