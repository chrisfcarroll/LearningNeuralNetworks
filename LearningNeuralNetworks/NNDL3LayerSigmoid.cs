using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LearningNeuralNetworks.Frameworks;
using LearningNeuralNetworks.Maths;
using LearningNeuralNetworks.V1;

namespace LearningNeuralNetworks
{
    static class np
    {
        static Random NewRandom() => new Random();

        public static double[][] random_randn(int size0, int size1)
        {
            var rnd = NewRandom();
            var result = new double[size0][];
            for (int i = 0; i < size0; i++)
            {
                result[i] = Enumerable.Range(0, size1).Select(j => rnd.NextDouble()).ToArray();
            }
            return result;
        }

        public static MatrixD dot(MatrixD left, MatrixD right) { return left * right; }
    }

    class NNDL3LayerSigmoid
    {
        readonly int layerCount;
        MatrixD[] biases;
        MatrixD[] weights;
        public TextWriter ConsoleOut = Console.Out;

        public NNDL3LayerSigmoid(int[] sizes)
        {
            /*
            """The list ``sizes`` contains the number of neurons in the
            respective layers of the network.  For example, if the list
            was [2, 3, 1] then it would be a three-layer network, with the
            first layer containing 2 neurons, the second layer 3 neurons,
            and the third layer 1 neuron.  The biases and weights for the
            network are initialized randomly, using a Gaussian
            distribution with mean 0, and variance 1.  Note that the first
            layer is assumed to be an input layer, and by convention we
            won't set any biases for those neurons, since biases are only
            ever used in computing the outputs from later layers.
            */
            layerCount = sizes.Length;

            //this.biases = [np.random_randn(y, 1) for y in sizes[1:]]
            biases = Enumerable.Range(1, layerCount).Select(y =>  (MatrixD)(np.random_randn(sizes[y], 1) )).ToArray();

            //this.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
            weights = Enumerable.Range(0, layerCount - 1).Select(i => (MatrixD)np.random_randn(sizes[i + 1], sizes[i])).ToArray();
        }

        MatrixD feedforward(MatrixD input)
        {
            /*Return the output of the network if ``a`` is input.*/
            //for b, w in zip(this.biases, this.weights): a = sigmoid(np.dot(w, a) + b)
            var a = input;
            biases.Zip(weights, (b, w) => { return a = (MatrixD) b + sigmoid(w*a); });
            return a;
        }

        public NNDL3LayerSigmoid SGD(IEnumerable<Pair<MatrixD, MatrixD>> training_data , int epochs , int mini_batch_size , double eta , IEnumerable<Pair<MatrixD, MatrixD>> test_data = null)
        {
            /*Train the neural network using mini-batch stochastic
            gradient descent.  The ``training_data`` is a list of tuples
            ``(x, y)`` representing the training inputs and the desired
            outputs.  The other non-optional parameters are
            this-explanatory.  If ``test_data`` is provided then the
            network will be evaluated against the test data after each
            epoch, and partial progress printed out.  This is useful for
            tracking progress, but slows things down substantially.*/

            // if test_data: n_test = len(test_data)
            test_data = test_data ?? new Pair<MatrixD, MatrixD>[0];
            var testCount = test_data.Count();
            var n = training_data.Count();
            var batchCount = n/mini_batch_size;

            //for j in range(epochs):
            for (int j = 0; j < epochs; j++)
            {
                // random.shuffle(training_data)
                // mini_batches = [ training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]

                var shuffledData = training_data.OrderRandomly();

                // for mini_batch in mini_batches:
                for (int b = 0; b < n; b+=mini_batch_size)
                {
                    var sample = shuffledData.Skip(b*mini_batch_size).Take(mini_batch_size);
                    update_mini_batch(sample, eta);
                }

                if (test_data.Any())
                {
                    ConsoleOut.WriteLine("Epoch {0}: {1} / {2}", j, evaluate(test_data), testCount);
                }
                else
                {
                    ConsoleOut.WriteLine("Epoch {0} complete", j);
                }
            }

            return this;
        }

        void update_mini_batch(IEnumerable<Pair<MatrixD,MatrixD>> mini_batch , double eta)
        {
            /*Update the network's weights and biases by applying
            gradient descent using backpropagation to a single mini batch.
            The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
            is the learning rate.*/

            // nabla_b = [np.zeros(b.shape) for b in self.biases]
            // nabla_w = [np.zeros(w.shape) for w in self.weights]
            var nabla_b = biases.Select(b => new MatrixD(b.RowCount, b.ColumnCount)).ToArray();
            var nabla_w = weights.Select(w => new MatrixD(w.RowCount, w.ColumnCount)).ToArray();
            var sampleSize = mini_batch.Count();

            foreach(var pair in mini_batch)
            {
                var x = pair.Data;
                var y = pair.Label;
                // delta_nabla_b, delta_nabla_w = self.backprop(x, y);
                // nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                // nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                var deltaNablas = backprop(x, y);
                nabla_b = nabla_b.Zip(deltaNablas.Item1, (nb, delta) => nb + delta).ToArray();
                nabla_w = nabla_w.Zip(deltaNablas.Item2, (nw, delta) => nw + delta).ToArray();
            }

            // self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
            // self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

            weights = weights.Zip(nabla_w, (w, nabla) => w - nabla * (eta / sampleSize)).ToArray();
            biases  = biases.Zip( nabla_b, (b, nabla) => b - nabla * (eta / sampleSize)).ToArray();
        }

        Tuple<MatrixD[],MatrixD[]> backprop(MatrixD input , MatrixD target )
        {
            /*Return a tuple ``(nabla_b, nabla_w)`` representing the
            gradient for the cost function C_x.  ``nabla_b`` and
            ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
            to ``this.biases`` and ``this.weights``.*/

            //nabla_b = [np.zeros(b.shape) for b in this.biases]
            //nabla_w = [np.zeros(w.shape) for w in this.weights] 

            var nabla_b = biases.Select(b => new MatrixD(b.RowCount, b.ColumnCount)).ToArray();
            var nabla_w = weights.Select(w => new MatrixD(w.RowCount, w.ColumnCount)).ToArray();

            // feedforward
            var activation = input;
            var activations = new List<double[][]>{input}; // list to store all the activations, layer by layer
            var zs = new List<double[][]>(); // list to store all the z vectors, layer by layer

            //for b, w in zip(this.biases, this.weights):
            foreach (var layer in biases.Zip(weights, (b, w) => new {Bias = b, Weight = w}))
            {
                var z = layer.Weight * activation + layer.Bias;
                zs.Add(z);
                activation = sigmoid(z);
                activations.Add(activation);
            }

            // backward pass
            //delta = this.cost_derivative(activations[-1], y)*sigmoid_prime(zs[-1])
            var delta = cost_derivative(activations.Last(), target) * sigmoid_prime(zs.Last());

            nabla_b[nabla_b.Length-1] = delta;
            nabla_w[nabla_w.Length-1] = delta * (activations.OrderByReverse().Skip(2).First()).AsMatrixD().T();

            // Note that the variable l in the loop below is used a little
            // differently to the notation in Chapter 2 of the book.  Here,
            // l = 1 means the last layer of neurons, l = 2 is the
            // second-last layer, and so on.  It's a renumbering of the
            // scheme in the book, used here to take advantage of the fact
            // that Python can use negative indices in lists.

            //for l in xrange(2, this.num_layers):
            for (int l = 2; l < layerCount; l++)
            {
                var z = zs.Last();
                delta = weights[weights.Length + 1 - l].T() * delta * sigmoid_prime(z);
                nabla_b[nabla_b.Length - l] = delta;
                nabla_w[nabla_b.Length - l] = delta * activations[-l - 1].AsMatrixD().T();
            }
            return Tuple.Create(nabla_b, nabla_w);
        }

        public int evaluate(IEnumerable<Pair<MatrixD,MatrixD>> test_data )
        {
            /*
            Return the number of test inputs for which the neural
            network outputs the correct result. Note that the neural
            network's output is assumed to be the index of whichever
            neuron in the final layer has the highest activation.*/
            
            // test_results = [ (np.argmax(this.feedforward(x)), y) for (x, y) in test_data]
            // return sum(int(x == y) for (x, y) in test_results)

            return test_data
                    .Select(p=> new {p.Label, Output= feedforward(p.Data)})
                    .Count(r => r.Label == r.Output);
        }

        /// <returns>The vector of partial derivatives ∂C(x)/∂a for the output activations</returns>
        MatrixD cost_derivative(MatrixD output_activations, MatrixD target) { return output_activations - target; }

        double sigmoid(double z) { return 1.0 / (1.0 + Math.Exp(-z)); }

        IEnumerable<double> sigmoid(IEnumerable<double> z) { return z.Select(sigmoid); }

        MatrixD sigmoid(double[][] z) { return z.Select(zi => zi.Select(sigmoid).ToArray()).ToArray(); }

        double sigmoid_prime(double z) { return sigmoid(z) * (1 - sigmoid(z)); }

        IEnumerable<double> sigmoid_prime(IEnumerable<double> z) { return z.Select(sigmoid_prime); }

        MatrixD sigmoid_prime(double[][] z) { return z.Select(zi=> zi.Select(sigmoid_prime).ToArray()).ToArray(); }
    }
}

