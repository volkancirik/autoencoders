import argparse

def get_parser_AE():
	parser = argparse.ArgumentParser()
	parser.add_argument('--lrate', action='store', dest='learning_rate',help='Learning Rate, default = 0.1',type=float,default = 0.1)

	parser.add_argument('--epochs', action='store', dest='n_epochs',help='# of epochs, default = 50',type=int,default = 50)

	parser.add_argument('--benchmark', action='store', dest='benchmark',help='benchmark name, default = mnistsmall.pkl.gz',default = 'mnistsmall.pkl.gz')

	parser.add_argument('--batch-size', action='store', dest='batch_size',help='batch size, default 20', type=int, default = 20)

	parser.add_argument('--v_size', action='store', dest='visible_size',help='visible size of input, default = 784',type=int,default = 784)

	parser.add_argument('--hidden', action='store', dest='hidden_size',help='hidden layer size, default = 500',type=int,default = 1000)

	parser.add_argument('--contraction-flag', action='store_true', dest='contraction_flag',help='Contractive AE {True | False} , default = False')

	parser.add_argument('--contraction-level', action='store', dest='contraction_level',help='contraction level, default = 0.1 (if contraction-flag is set)',type=float,default = 0.1)

	parser.add_argument('--sparse-flag', action='store_true', dest='sparse_flag',help='Sparse AE {True | False} , default = False')

	parser.add_argument('--sparsity-penalty', action='store', dest='sparsity_penalty',help='sparsity penalty, default = 0.001 (if sparsity-flag is set)',type=float,default = 0.001)

	parser.add_argument('--sparsity-level', action='store', dest='sparsity_level',help='sparsity level, default = 0.05 (if sparsity-flag is set)',type=float,default = 0.05)

	parser.add_argument('--corruption-rate', action='store', dest='corruption_rate',help='corruption rate, default = 0.0',type=float,default = 0.0)

	parser.add_argument('--activation', action='store', dest='activation',help='activation function {sigmoid,tanh,softplus}, default = sigmoid',default = "sigmoid")

	parser.add_argument('--cost-type', action='store', dest='cost_type',help='cost type {MeanSquaredCost,CrossEntropy,CategoricalCrossEntropy}, default = MeanSquaredCost',default = "MeanSquaredCost")

	parser.add_argument('--output-fname', action='store', dest='fname',help='output filename, default = ae with parameter details appended +.pkl',default = "ae")

	parser.set_defaults(contraction_flag = False)

	parser.set_defaults(sparse_flag = False)

	return parser
