import sys
import random
import shutil

if __name__ == "__main__":
	input_file = sys.argv[1]
	output_file = sys.argv[2]
	f_in = open(input_file, 'r')
	f_out = open(output_file, 'w+')

	line = f_in.readline()
	line = f_in.readline()
	num = line[1:].split()
	f_out.write('%%MatrixMarket matrix coordinate pattern general\n')
	f_out.write(str(num[1]) + ' ')
	f_out.write(str(num[2]) + ' ')
	f_out.write(str(num[0]) + '\n')
	
	shutil.copyfileobj(f_in, f_out)

	f_in.close()
	f_out.close()