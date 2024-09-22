import os

#读取字符表
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

#列出所有图片
def list_file_img(path):
	file_list = []

	for file in os.listdir(path):
		if not os.path.isdir(file):
			end_pos = file.rfind('.')
			suffix = file[end_pos + 1:]
			if suffix == 'jpg':
				prefix = file[:end_pos]
				file_list.append(prefix)
		else:
			file_list.extend(list_file_img(file))

	return file_list


if __name__ == '__main__':
	chars_file = 'chars.txt'
	chars_map = read_chars_map(chars_file)

	#读取所有图片若有新则加入字符表
	files = list_file_img('./')
	for file_name in files:
		for c in file_name:	
			try:
				chars_map.index(c)
			except:
				chars_map.append(c)
	
	#for i in range(len(chars_map)):
	#	chars_map[i] = chars_map[i] + '\n'
	

	print(chars_map)

	f = open(chars_file, 'w')
	
	f.seek(0,0)
	
	for item in chars_map:
		f.write(item + '\n')

	f.close()

