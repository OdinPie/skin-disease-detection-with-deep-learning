import streamlit as st
import numpy as np
import tensorflow as tf

def model_prediction(test_image):
    try:
        my_model = tf.keras.models.load_model('my_trained_model.h5')
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

    try:
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(224,224))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        prediction = my_model.predict(input_arr)
        result_index = np.argmax(prediction)
        if result_index == 0:
            treatment = "Topical Corticosteroids: Used to reduce inflammation and itching.\
                        \n Moisturizers: Essential to keep the skin hydrated.\
                        \n Antihistamines: Help to control severe itching.\
                        \nImmunomodulators: Topical calcineurin inhibitors like tacrolimus and pimecrolimus.\
                        \nPhototherapy: Ultraviolet light therapy can be effective for severe cases."
        elif result_index == 1:
            treatment = "\nCryotherapy: Freezing the cancer cells with liquid nitrogen.\
                        \nTopical Treatments: Such as imiquimod and 5-fluorouracil (5-FU) for superficial BCC.\
                        \nRadiation Therapy: Used when surgery is not an option."
        elif result_index == 2:
            treatment = "\nCryotherapy: Freezing the cancer cells with liquid nitrogen.\
                        \nLaser Therapy: Using laser to remove the lesions.\
                        \nTopical Treatments: Such as retinoids."
        elif result_index == 3:
            treatment = "\nTopical Corticosteroids: To reduce inflammation.\
                        \nMoisturizers: Regular use to maintain skin hydration.\
                        \nCalcineurin Inhibitors: Tacrolimus and pimecrolimus for long-term use."
        elif result_index == 4:
            treatment = "\nObservation: Regular monitoring for changes in size, color, or shape.\
                        \nSurgical Removal: If changes are suspicious or for cosmetic reasons."
        elif result_index == 5:
            treatment = "\nSurgical Excision: Wide local excision of the tumor with a margin of normal skin.\
                        \nImmunotherapy: Drugs like pembrolizumab or nivolumab to boost the immune response.\
                        \nTargeted Therapy: For melanomas with specific mutations (e.g., BRAF inhibitors)."
        elif result_index == 6:
            treatment = "\nTopical Treatments: Corticosteroids, retinoids, or calcineurin inhibitors.\
                        \nPhototherapy: UVB light therapy.\
                        \nOral Medications: Antihistamines to relieve itching."
        elif result_index == 7:
            treatment = "\nCryotherapy: Freezing with liquid nitrogen.\
                        \nCurettage: Scraping off the lesion.\
                        \nElectrosurgery: Using electrical current to remove the lesions."
        elif result_index == 8:
            treatment = "\nTopical Antifungals: Clotrimazole, miconazole for mild infections.\
                        \nOral Antifungals: For more extensive infections, such as terbinafine or fluconazole.\
                        \nKeeping the Area Dry and Clean: To prevent the spread of the infection."
        else:
            treatment = "\nCryotherapy: Freezing the cancer cells with liquid nitrogen.\
                        \nCantharidin: A blistering agent applied by a healthcare provider.\
                        \nLaser Treatment: For resistant warts."
        
        return result_index,treatment
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None


class_name = [
 'Atopic Dermatitis' ,
 'Basal Cell Carcinoma (BCC)',
 'Benign Keratosis-like Lesions (BKL)',
 'Eczema',
 'Melanocytic Nevi (NV)',
 'Melanoma',
 'Psoriasis pictures Lichen Planus and related diseases',
 'Seborrheic Keratoses and other Benign Tumors',
 'Tinea Ringworm Candidiasis and other Fungal Infections',
 'Warts Molluscum and other Viral Infections'
]

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Detection"])

#Home Page
if(app_mode == "Home"):
    st.markdown("<h1 style='text-align: center;'>SKIN DISEASE DETECTION SYSTEM</h1>", unsafe_allow_html=True)
    # st.header("SKIN DISEASE DETECTION SYSTEM")
    image_path = "asset\home.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    # Welcome to the Skin Disease Detection System! 💉👩‍⚕️🔍

---

## Our Mission

Skin diseases are a challenging group of conditions that can be categorized into those that are commonly seen and easily recognized based on the characteristic site of presentation, size, color, distribution, and symptoms. Understanding the different types of skin diseases and their characteristics is crucial for accurate diagnosis and treatment.
Skin diseases are classified into the following 10 categories:

1. **Eczema**
   

2. **Melanoma**
   

3. **Atopic Dermatitis**
   

4. **Basal Cell Carcinoma (BCC)**
   

5. **Melanocytic Nevi (NV)**
   

6. **Benign Keratosis-like Lesions (BKL)**
   

7. **Psoriasis pictures Lichen Planus and related diseases**
   

8. **Seborrheic Keratoses and other Benign Tumors**
   

9. **Tinea Ringworm Candidiasis and other Fungal Infections**
   

10. **Warts Molluscum and other Viral Infections**
    

Each category of skin diseases presents unique characteristics and challenges in diagnosis and treatment. By categorizing them based on common features such as the site of presentation, size, color, distribution, and symptoms, healthcare professionals can more easily identify and manage these conditions.

## How It Works

Our platform uses state-of-the-art deep learning models to analyze uploaded images of skin conditions and predict the potential disease. Here's a quick overview of how it works:

1. **Upload an Image**: Simply upload a clear image of the affected skin area.
2. **AI Analysis**: Our AI model processes the image and runs a series of diagnostics.
3. **Get Results**: Receive a detailed report on the detected skin condition along with possible treatment options.


## Why Choose Us?

### **Accuracy**

Our AI models are trained on hundreds of images to ensure high accuracy in detecting various skin diseases, from common conditions like eczema to more serious concerns like melanoma.

### **Ease of Use**

Our platform is designed with user experience in mind. The interface is simple and straightforward, making it easy for anyone to use, regardless of technical skill.

### **Comprehensive Database**

We cover a wide range of skin conditions, ensuring that you receive accurate diagnostics for numerous skin issues.

### **Security and Privacy**

We prioritize your privacy. All uploaded images and personal data are securely stored and processed, ensuring that your information remains confidential.

---

Get started today and take the first step towards better skin health with **Skin Disease Detection**!


    """)


#About Page
elif(app_mode=="About"):
    st.markdown("<h1 style='text-align: center;'>About Us</h1>", unsafe_allow_html=True)
    st.markdown("""
                #### About Dataset
                This dataset consists of about 27.2K rgb images of  diseased skin which is categorized into 10 different classes.The total dataset is divided into 80/10 ratio of training and validation set preserving the directory structure.
                A new directory containing 564 test images is created later for prediction purpose.
                #### Content
                1. train (446 images)
                2. test (66 images)
                3. validation (52 images)

                #### The dataset was created by Ismail Hossain on Kaggle
                Link: https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset/data            
    """)
#Detection Page
elif(app_mode=="Disease Detection"):
    # st.header("Disease Detection")
    st.markdown("<h1 style='text-align: center; color: blue;'>Disease Detection</h1>", unsafe_allow_html=True)
    test_image = st.file_uploader("Choose an Image")
    if(st.button("Show Image")):
        st.image(test_image,use_column_width=True)
    if(st.button("Detect")):
        st.write("Our Detected Disease: ")
        with st.spinner('Processing...'):
                result_index,treatment = model_prediction(test_image)
                if result_index is not None:
                    st.success("Model is Predicting it's a {}".format(class_name[result_index]))
                    st.write("Recommended Treatment: ")
                    st.info(treatment)


