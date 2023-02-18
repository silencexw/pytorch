import numpy as np
import torch.utils.data as data
from PIL import  Image
from torchvision import transforms

def preprocess_input(x):
    x/=127.5
    x-=1.
    return x
def cvtColor(image):
    if len(np.shape(image))==3 and np.shape(image)[-2]==3:
        return image
    else:
        image=image.convert('RGB')
        return image


class DataGenerator(data.Dataset):
    def __init__(self,annotation_lines,inpt_shape,random=True):
        self.annotation_lines=annotation_lines
        self.input_shape=inpt_shape
        self.random=random

    def __len__(self):
        return len(self.annotation_lines)
    def __getitem__(self, index):
        annotation_path=self.annotation_lines[index].split(';')[1].split()[0]
        image=Image.open(annotation_path)
        #image=self.get_random_data(image,self.input_shape,random=self.random)
        # resize = transforms.Resize([224, 224])
        # image= resize(image)
        image=np.array(image)
        image=np.transpose(preprocess_input(image.astype(np.float32)),[2,0,1])
        y=int(self.annotation_lines[index].split(';')[0])
        return image,y
    def rand(self,a=0,b=1):
        return np.random.rand()*(b-a)+a
