from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
import argparse
from six.moves import range
from parse import parse

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("text", help="text to be parsed")
	parser.add_argument("output", help="output parsed file")
	parser.add_argument("model", help="trained model file")
	parser.add_argument("schema", help="document schema file")
	args = parser.parse_args()
	text = args.text
	output = args.output
	model = args.model
	(tokens,mentions) = parse(text,output,model)
	print("The tokens in the document are",tokens)
	print("The entities in the document are",mentions)
