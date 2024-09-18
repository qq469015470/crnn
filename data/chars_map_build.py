import os

def read_chars_map(chars_file):
	chars_map = []
	
	f = open(chars_file, 'a+', encoding='utf8')
	f.seek(0, 0)
	for line in f.read():
		if line != '\n':
			try:
	        		chars_map.index(line)
			except:
				chars_map.append(line)
	
	f.close()
	
	return chars_map


if __name__ == '__main__':
	chars_file = 'chars.txt'
	chars_map = read_chars_map(chars_file)
	print(chars_map)

	for file in os.listdir('./'):
		if not os.path.isdir(file):
			end_pos = file.rfind('.')
			suffix = file[end_pos + 1:]
			if suffix == 'jpg':
				prefix = file[:end_pos]
				for c in prefix:	
					try:
						chars_map.index(c)
					except:
						chars_map.append(c)
	
	
	#for i in range(len(chars_map)):
	#	chars_map[i] = chars_map[i] + '\n'
	
	f = open(chars_file, 'w')
	
	f.seek(0,0)
	
	for item in chars_map:
		f.write(item + '\n')

	f.close()

