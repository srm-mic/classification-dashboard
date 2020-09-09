#imports
import streamlit as st
from torchvision import models, transforms
import torch
from PIL import Image

#command to stop file encoder warning
st.set_option('deprecation.showfileUploaderEncoding', False)

#storing models in cache
@st.cache(allow_output_mutation=True)
def dnet121():
    densenet121 = models.densenet121(pretrained=True)
    return densenet121
@st.cache(allow_output_mutation=True)
def dnet161():
    densenet161 = models.densenet161(pretrained=True)
    return densenet161
@st.cache(allow_output_mutation=True)
def rnet34():
    resnet34 = models.resnet34(pretrained=True)
    return resnet34   
@st.cache(allow_output_mutation=True)
def rnet50():
    resnet50 = models.resnet50(pretrained=True)
    return resnet50     
@st.cache(allow_output_mutation=True)
def gnet():
    googlenet = models.googlenet(pretrained=True)
    return googlenet   
@st.cache(allow_output_mutation=True)
def inetv3():
    inception = models.inception_v3(pretrained=True)
    return inception

#function to get name from the predicted class
def name(pred):
    name = ""
    for char in pred:
        if ord(char) >= 65 and ord(char) <= 90:
            name += char
        elif ord(char) >=97 and ord(char) <= 122:
            name += char
        elif ord(char) == 32:
            name += char
    return name

#function to show image
def showimg(x):
    st.image(x,use_column_width=True)
    
#main function
def main():

    #title and header
    st.title("Classification Dashboard")
    st.subheader("A cool dashboard to run inference on pretrained classification models on streamlit.")
      
    #widget to upload and display image
    st.subheader("Upload image you want to be classified here")
    upl = st.file_uploader('')

    if st.button("View"):
       
        showimg(upl)
    
    #transforming the image
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )])    
    
    #choosing the model to classify
    st.subheader("Which classification model do you want to use:")
    modelname = st.selectbox('',['DenseNet121','DenseNet161','GoogLeNet','InceptionNet','ResNet34','ResNet50'])
    if st.button("Classify"):
        img = transform(Image.open(upl))
        img = torch.unsqueeze(img,0)

        #loading the selected model
        if modelname == 'DenseNet121':
            model = dnet121()
        elif modelname == 'DenseNet161':
            model = dnet161()
        elif modelname == 'GoogLeNet':
            model = gnet()
        elif modelname == 'InceptionNet':
            model = inetv3()
        elif modelname == 'ResNet34':
            model = rnet34()
        elif modelname == 'ResNet50':
            model = rnet50()
        
        #evaluating the image
        model.eval()
        output = model(img)

        #loading ImageNet classes
        with open('simple_imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]

        #Classifying
        prob = torch.nn.functional.softmax(output, dim=1)[0] * 100
        _, index = torch.max(output, 1)
        print(classes[index[0]], prob[index[0]].item())

        #Removing numbers and special characters and outputting result
        pred = classes[index[0]]
        predname = name(pred)
        st.subheader("Your image is classified as **%s** with a probability of **%.2f** using **%s**"%
            (predname, prob[index[0]], modelname))
        
    
    
if __name__ == '__main__':
    main()