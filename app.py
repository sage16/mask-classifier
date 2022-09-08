import io
import torchvision
import streamlit as st
import torch
from PIL import Image


class_names = ['with_masked', 'without_mask']


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def load_predict(image):
    with open('mask_classifier.pt', 'rb') as f:
        checkpoint = torch.load(f)

    weight = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weight)

    output_shape = len(class_names)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=output_shape,
                        bias=True))
    model.load_state_dict(checkpoint)

    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()
    img_trans = transforms(image).unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(model(img_trans), dim=1)
        pred_label = torch.argmax(pred_probs, dim=1)
        pred_class = class_names[pred_label]
    print(pred_class)
    st.write(f'The picture is a person {pred_class}')


def main():
    st.title('Mask Classifier WebApp')
    st.write('This app takes in a picture and tells if the person in the picture is masked or unmasked')
    image = Image.open('mask_unmask.jpg')
    st.image(image, use_column_width=True)
    pic = load_image()
    result = st.button('Run on Image')
    if result:
        st.write('Calculating result...')
        load_predict(pic)


if __name__ == '__main__':
    main()
