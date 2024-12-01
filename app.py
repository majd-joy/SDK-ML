import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

try:
    model = tf.keras.models.load_model(r'C:\Users\NTC\Desktop\glami\fashion_mnist_model.h5')
    st.success('Model loaded successfully')
except Exception as e:
    st.error(f'Error loading model: {e}')

def prepare_image(img):
    try:
        img = img.resize((28, 28)) 
        img = img.convert('L')  
        img = np.array(img)
        img = img.flatten()  
        img = img / 255.0  
        img = np.expand_dims(img, axis=0)  
        return img
    except Exception as e:
        st.error(f'Error processing image: {e}')
        return None


st.title('GLAMIFY')

uploaded_file = st.file_uploader("Choose an image of your clothing", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        prepared_img = prepare_image(image)
        if prepared_img is not None:
            predictions = model.predict(prepared_img)
            predicted_class = np.argmax(predictions[0])
            
            classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            class_name = classes[predicted_class]

            recommendations = {
                'T-shirt/top': 'Pair it with jeans and sneakers.',
                'Trouser': 'Match it with a smart shirt.',
                'Pullover': 'Wear it over a casual shirt.',
                'Dress': 'Perfect with sandals or heels.',
                'Coat': 'Ideal with winter boots.',
                'Sandal': 'Great with shorts or a summer dress.',
                'Shirt': 'Suitable for both formal and casual outfits.',
                'Sneaker': 'Pair it with sportswear or casual outfits.',
                'Bag': 'Accessorize with matching shoes.',
                'Ankle boot': 'Looks great with skinny jeans or leggings.'
            }

            recommendation = recommendations[class_name]

            st.write(f'**Clothing Item:** {class_name}')
            st.write(f'**Recommendation:** {recommendation}')
    except Exception as e:
        st.error(f'Error processing image: {e}')




