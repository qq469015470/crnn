import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms as transforms
import random
import PIL

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
print(chars_map)

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

		self.lstm = torch.nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
		self.prev_hidden = torch.zeros(1, 1, 256)
		self.c = torch.zeros(1,1, 256)
		self.linear = torch.nn.Linear(256, len(chars_map))

	def forward(self, x):#接收(batch,1,W,32)灰度图片(图片长宽比不变缩放到高度32)
		output = self.conv1(x)
		output = self.conv2(output)
		output = self.conv3(output)
		output = self.conv4(output)
		output = self.conv5(output)
		#输出(batch,channel, 1, width)

		#print(output.size())
		output = output.squeeze(2)
		#print(output.size())
		output = output.reshape(-1, output.size(2), output.size(1))
		#print(output.size())
		output, _ = self.lstm(output, (self.prev_hidden, self.c))	
		#self.prev_hidden = prev_hidden.detach()
		

		output = self.linear(torch.relu(output))

		output = torch.nn.functional.log_softmax(output, dim=2)


		output = output.reshape(output.size(1), output.size(0), output.size(2))

		return output

def str_to_label(val):
	target = []
	for item in val:
		target.append(chars_map.index(item))

	return torch.tensor(target)

def read_train_data(file_name):
	data_path = './data/'
	img = mpimg.imread(data_path + file_name)
	
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
	

def train():
	EPOCH=10000
	optimizer = optim.Adam(model.parameters(), 0.01)
	criterion= torch.nn.CTCLoss(blank=0, reduction='mean')

	for i in range(EPOCH):
		target_img = img[int(random.random() * 3)]
		output = model(target_img['data'])
		#print(type(target.size(1)))

		target = target_img['label'].unsqueeze(0)

		loss = criterion(output, target, torch.IntTensor([output.size(0)]), torch.IntTensor([target.size(1)]))
		model.zero_grad()
		loss.backward()
		optimizer.step()

		if i % 10 == 0:
			s = ''
			for j in range(output.size(0)):
				s += chars_map[torch.argmax(output[j][0]).item()]

			print(s)
			#print(chars_map[torch.argmax(output[72][0]).item()])
			print('loss:%f' % (loss))



img = [
	{ 'label' : str_to_label('HH4尤'), 'data': read_train_data('./HH4尤.jpg') },
	{ 'label' : str_to_label('4尤'), 'data': read_train_data('./4尤.jpg') },
	{ 'label' : str_to_label('3'), 'data': read_train_data('./3.jpg') },
]

model = CRNN()

print(str_to_label('HH4尤'))
train()
