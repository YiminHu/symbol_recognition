import operation
import os
dirname = './seed/'
listdir = os.listdir(dirname)
for filename in listdir:
	    if not filename.startswith('.'):
		    operation.gen_from_seed(dirname+filename)
