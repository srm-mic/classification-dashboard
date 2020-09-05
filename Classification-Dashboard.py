#imports
import streamlit as st
from torchvision import models, transforms
import torch
from PIL import Image

#command to stop file encoder warning
st.set_option('deprecation.showfileUploaderEncoding', False)

#storing models in cache
@st.cache(allow_output_mutation=True)
def dnet():
    densenet = models.densenet121(pretrained=True)
    return densenet
@st.cache(allow_output_mutation=True)
def rnet():
    resnet = models.resnet50(pretrained=True)
    return resnet
@st.cache(allow_output_mutation=True)
def inet():
    inception = models.inception_v3(pretrained=True)
    return inception
    
#main function
def main():

    #title and header
    st.title("Classification Dashboard")
    st.subheader("A cool dashboard to run inference on pretrained classification models on streamlit.")
      
    #widget to upload and display image
    st.subheader("Upload image to classified here")
    upl = st.file_uploader('')
    image = Image.open(upl).convert('RGB')

    if st.button("View"):
       
        st.image(image,use_column_width=True)
    
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
    model = st.selectbox('',['AlexNet','Vgg16','ResNet'])
    if st.button("Classify"):
        img = torch.unsqueeze(transform(image),0)

     
    if model == 'ResNet':
       resnet.eval()
       output = resnet(img)
    #elif model == 'Vgg16':
       # mod = vgg
    
    #loading ImageNet classes
    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    #Classifying
    prob = torch.nn.functional.softmax(output, dim=1)[0] * 100
    _, index = torch.max(output, 1)
    print(classes[index[0]], prob[index[0]].item())
    st.subheader("Your image is classified as")
    st.subheader(classes[index[0]])
    
    
if __name__ == '__main__':
    main()