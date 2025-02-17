"""
    Running this code in Spyder will launch a web page on the Streamlit platform.
"""
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from PIL import Image
import io
import os
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Skin Cancer Detection System", layout="wide")

#loading the saved model
loaded_model = pickle.load(open("C:/Users/thany/OneDrive/Desktop/MTech/SET_project/trained_model.sav", 'rb'))


#creating the function for prediction
def skinDiseasePrediction(input_img):
    types = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    img = image.load_img(input_img, target_size=(224, 224))
    img_array = image.img_to_array(img)
    x = preprocess_input(img_array)

    prediction = loaded_model.predict(np.array([x]))[0]
    test_pred = np.argmax(prediction)

    result = [(types[i], float(prediction[i]) * 100.0) for i in range(len(prediction))]
    result.sort(reverse=True, key=lambda x: x[1])
    
    st.write("Prediction Results:")
    for j in range(7):
        (class_name, prob) = result[j]
        st.write(f"Top {j + 1} ====================")
        st.write(f"{class_name}: {prob:.2f}%")

    
def main():
    st.title("Skin Cancer Detection System")
    st.markdown(
    """
    <style>
    /* Hide the collapse (hamburger) button in the sidebar */
    [data-testid="collapsedControl"] {
        display: none;
    }

    /* Improved Card Styling */
    .card {
        background-color: white;  /* White background for better contrast */
        border: 2px solid #ddd;  /* Slightly darker border for visibility */
        padding: 20px;
        border-radius: 8px;
        margin: 10px;
        box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.1);  /* Soft shadow effect */
        color: black;  /* Ensures text is readable */
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    if 'page' not in st.session_state:
        st.session_state.page = "Home"

    st.sidebar.title("Menu")
    # Using buttons for navigation:
    if st.sidebar.button("Home"):
        st.session_state.page = "Home"
    if st.sidebar.button("Predict"):
        st.session_state.page = "Predict"    
    if st.sidebar.button("Chatbot"):
            st.session_state.page = "Chatbot"
    
    if st.session_state.page == "Home":
        st.header("Skin Cancer Information")
    
        # List of skin cancer types with descriptions.
        skin_cancers = [
            {
                "title": "Actinic Keratosis (akiec)",
                "description": "Actinic Keratosis (akiec) is a precancerous skin condition caused by long-term exposure to the sun. It appears as rough, scaly patches on the skin, commonly found on the face, scalp, ears, and hands. If left untreated, actinic keratosis can develop into squamous cell carcinoma, a type of skin cancer. These lesions are often red or brown, with a sandpaper-like texture."
                },
            {
                "title": "Basal Cell Carcinoma (bcc)",
                "description": "Basal Cell Carcinoma (bcc) is the most common form of skin cancer, but it grows slowly and rarely spreads to other parts of the body. It often appears as a pearly or waxy bump with visible blood vessels, commonly on sun-exposed areas such as the face and neck. Although not highly aggressive, bcc can cause significant local tissue damage if left untreated."
                },
            {
                "title": "Benign Keratosis-like Lesions (bkl)",
                "description": "Benign Keratosis-like Lesions (bkl) are non-cancerous skin growths that can resemble malignant tumors. They include conditions such as seborrheic keratosis and solar lentigines, which typically appear as brown, rough, or scaly patches. Although harmless, these lesions can sometimes be mistaken for more serious conditions like melanoma."
                },
            {
                "title": "Melanoma (mel)",
                "description": "Melanoma (mel) is the most dangerous type of skin cancer, originating from melanocytes, the pigment-producing cells in the skin. It often appears as an irregularly shaped, dark-colored mole with uneven borders. If not detected early, melanoma can spread rapidly to other parts of the body, making early diagnosis and treatment crucial."
                },
            {
                "title": "Dermatofibroma (df)",
                "description": "Dermatofibroma (df) is a benign skin tumor that consists of fibrous tissue. It usually appears as a firm, small, dark brown or reddish nodule on the skin, often found on the legs. Dermatofibromas are harmless and do not require treatment unless they become bothersome or painful."
                },
            {
                "title": "Melanocytic Nevus (nv)",
                "description": "Melanocytic Nevus (nv) refers to a common mole, which is a benign growth of melanocytes. Moles can appear anywhere on the skin and vary in color from light brown to black. While most melanocytic nevi are harmless, some can develop into melanoma over time, especially if they change in size, shape, or color."
                },
            {
                "title": "Vascular Lesions (vasc)",
                "description": "Vascular Lesions (vasc) include various types of blood vessel abnormalities, such as angiomas, hemorrhages, and vascular malformations. These lesions often appear as red or purple spots on the skin and are usually benign. Although vascular lesions are generally harmless, some may require treatment for cosmetic reasons or if they cause discomfort."
                }
            ]
    
        for i in range(0, len(skin_cancers), 3):
            cols = st.columns(3)
            for col, cancer in zip(cols, skin_cancers[i:i+3]):
                with col:
                    st.markdown(f"""
                                <div class="card">
                                <h3>{cancer['title']}</h3>
                                <p>{cancer['description']}</p>
                                </div>
                                """, unsafe_allow_html=True)
    
    if st.session_state.page == "Predict":        
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        prediction = ""
    
        if st.button("Detect"):
            if uploaded_image is None:
                st.warning("Please upload an image before detection!")
            else:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", width=300)
                prediction = skinDiseasePrediction(uploaded_image)
    elif st.session_state.page == "Chatbot":
        st.header("Glow AI")
        st.write("Welcome to the Glow AI Chatbot. Type your message below to start a conversation.")
    
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
        # Text input for the user's message
        user_message = st.text_input("Your message:", key="chat_input")
    
        # "Send" button to submit the message and update chat history
        if st.button("Send"):
            if user_message:
                st.session_state.chat_history.append(("User", user_message))
                # Placeholder response – replace with your actual chatbot logic
                bot_response = "This is a placeholder response."
                st.session_state.chat_history.append(("Bot", bot_response))
                
                # Display the chat history
                st.markdown("### Chat History")
                for speaker, message in st.session_state.chat_history:
                    if speaker == "User":
                        st.markdown(f"**You:** {message}")
                    else:
                        st.markdown(f"**Bot:** {message}")            

if __name__=='__main__':
    main()
