import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms as transforms
import random
import PIL
import os
import sys

#读取字符表文件
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


#img = mpimg.imread('./tupian.jpg')

#plt.imshow(img)

#ax = plt.gca()

#ax.add_patch(plt.Rectangle((10,5), 10, 10, color='blue', fill=False, linewidth=1))

#plt.show()

#chars_map = ['_', 'H', '4', '尤']
chars_map = read_chars_map('./data/chars.txt')
chars_map.insert(0, '_')
print('chars_map: ', chars_map)

class CRNN(torch.nn.Module):
	def __init__(self):
		super(CRNN, self).__init__()
		self.conv1 = torch.nn.Sequential(
			torch.nn.Conv2d(1, 64, kernel_size=3, padding=1),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=2),
		)	

		self.conv2 = torch.nn.Sequential(
			torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=2),
		)	

		self.conv3 = torch.nn.Sequential(
			torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
			torch.nn.ReLU(),
			torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
		)	

		self.conv4 = torch.nn.Sequential(
			torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
		)

		self.conv5 = torch.nn.Sequential(
			torch.nn.Conv2d(512, 512, kernel_size=2, stride=1),
			torch.nn.ReLU(),
		)	

		self.lstm = torch.nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True, bidirectional=True)
		self.prev_hidden = torch.zeros(2, 1, 512)
		self.c = torch.zeros(2, 1, 512)
		self.linear = torch.nn.Linear(1024, len(chars_map))

	def forward(self, x):#接收(batch,1,32,W)灰度图片(图片长宽比不变缩放到高度32)
		output = self.conv1(x)
		output = self.conv2(output)
		output = self.conv3(output)
		output = self.conv4(output)
		output = self.conv5(output)#输出(batch,channel, 1, width)

		#转换成(batch, width, channel)
		output = output.squeeze(2)
		cnn_x = output.reshape(-1, output.size(2), output.size(1))
		#====

		output, _ = self.lstm(cnn_x, (self.prev_hidden, self.c))	

		#全连接层
		output = self.linear(torch.relu(output))

		#然后softmax
		output = torch.nn.functional.log_softmax(output, dim=2)


		#转换成(seq, batch, y)
		output = output.reshape(output.size(1), output.size(0), output.size(2))

		return output

#字符转成数据化的值
def str_to_label(val):
	target = []
	for item in val:
		target.append(chars_map.index(item))

	return torch.tensor(target)

#数据化的值转换成字符
def label_to_str(label):
	s = chars_map[torch.argmax(label).item()]
	return s

#读取训练图片(处理图片按比例resize成高度32px)
def read_train_data(file_name):
	img = mpimg.imread(file_name)
	
	transform = transforms.Compose([
	   transforms.Grayscale(1), #这一句就是转为单通道灰度图像
	   transforms.ToTensor(),
	   transforms.Normalize(mean = [0.5], std = [0.5]),
	])

	img = PIL.Image.fromarray(img, mode='RGB')
	(w,h) = img.size
	ratio = h / 32
	img = img.resize((int(w * ratio), 32))
	img = transform(img)
	img = img.unsqueeze(0)

	return img

#列出所有图片路径及名称
def list_file_img(path):
	file_list = []

	for file in os.listdir(path):
		if not os.path.isdir(path + file):
			end_pos = file.rfind('.')
			suffix = file[end_pos + 1:]
			if suffix == 'jpg':
				file_list.append((path, file))

		else:
			file_list.extend(list_file_img(path + file + '/'))

	return file_list
	
#获取所有训练图片
def get_all_target():
	data_path = './data/'

	img = []
	for (path, file) in list_file_img(data_path):
		end_pos = file.rfind('.')	
		suffix = file[end_pos + 1:]
		if suffix == 'jpg':
			prefix = file[:end_pos]
			img.append({
				'label_str': prefix,
				'label': str_to_label(prefix),
				'file': path + file,  
			})
					
	return img	


def train():
	EPOCH=20000
	optimizer = optim.Adam(model.parameters(), 0.01)
	criterion= torch.nn.CTCLoss(blank=chars_map.index('_'), reduction='mean')

	for i in range(EPOCH):
		target_img = img[int(random.random() * len(img))]
		output = model(read_train_data(target_img['file'])) #(seq,batch,word_vec)
		#print(type(target.size(1)))

		target = target_img['label'].unsqueeze(0)

		loss = criterion(output, target, torch.IntTensor([output.size(0)]), torch.IntTensor([target.size(1)]))
		model.zero_grad()
		loss.backward()
		optimizer.step()

		if i % 10 == 0:
			out_str = ''
			for j in range(output.size(0)):
				out_str += label_to_str(output[j][0])

			print('target:', target_img['label_str'])
			print('prep:', out_str)
			#print(chars_map[torch.argmax(output[72][0]).item()])
			print('loss:%f' % (loss))


model = CRNN()
model.load_state_dict(torch.load('bi_checkpoint.pth')['model']);#读取权重

if len(sys.argv) == 1:
	img = get_all_target()
	#print(img)
	#img = [
	#	{ 'label' : str_to_label('HH4尤'), 'data': read_train_data('./HH4尤.jpg') },
	#	{ 'label' : str_to_label('4尤'), 'data': read_train_data('./4尤.jpg') },
	#	{ 'label' : str_to_label('3'), 'data': read_train_data('./3.jpg') },
	#]
	train()
	torch.save({'model': model.state_dict()}, './bi_checkpoint.pth');#保存训练好的权重
elif sys.argv[1] == 'test':
	file_path = sys.argv[2]
	test_img = read_train_data(file_path)
	plt.imshow(test_img.squeeze(0).squeeze(0))
	plt.show()
	str_label = file_path[file_path.rfind('/') + 1:file_path.rfind('.')]

	print('测试标签:' + str_label)

	output = model(test_img)	
	out_str = ''
	for i in range(output.size(0)):
		out_str += label_to_str(output[i][0])

	print(out_str)

else:
	print('参数错误')
