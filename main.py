import streamlit as st
import numpy as np
from PIL import Image
from patchify import patchify
from tensorflow.keras.models import load_model

# Set page configuration
st.set_page_config(
    page_title="mDELRelapseNet",
    page_icon="🔬",
    layout="wide"
)


# Custom CSS
def local_css():
    st.markdown("""
        <style>
        .main {
            padding: 2rem 3rem;
        }
        .stTitle {
            color: #2c3e50;
            font-size: 3rem !important;
            padding-bottom: 1rem;
        }
        .stSubheader {
            color: #34495e;
            font-size: 1.5rem !important;
        }
        .stButton button {
            background-color: #2980b9;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 2rem;
        }
        .stButton button:hover {
            background-color: #3498db;
        }
        .css-1d391kg {
            padding: 2rem;
            border-radius: 10px;
            background-color: #f8f9fa;
        }
        </style>
    """, unsafe_allow_html=True)

def combine_images(img1, img2, img3):
    img1 = np.array(img1.convert('L'))
    img2 = np.array(img2.convert('L'))
    img3 = np.array(img3.convert('L'))

    max_intensity = max(np.max(img1), np.max(img2), np.max(img3))
    img1 = (img1 / np.max(img1)) * max_intensity
    img2 = (img2 / np.max(img2)) * max_intensity
    img3 = (img3 / np.max(img3)) * max_intensity

    combined_img = np.stack([img1, img2, img3], axis=-1).astype(np.uint8)
    return combined_img

def create_patches(img_array, patch_size=448):
    patches = patchify(img_array, (patch_size, patch_size, 3), step=patch_size)
    patch_list = []

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i, j, 0]
            patch_img = Image.fromarray(patch.astype(np.uint8))
            patch_list.append(patch_img)
    
    return patch_list

def is_valid_patch(patch_img, black_threshold=0.95):
    gray_patch = np.array(patch_img.convert('L'))
    black_ratio = np.sum(gray_patch < 50) / gray_patch.size
    return black_ratio <= black_threshold

def load_appropriate_model(image_type, cell_origin):
    model_path = ''
    if image_type == 'mfIHC':
        if cell_origin == 'unknown':
            model_path = 'models/both/mfIHC/model-best.h5'
        elif cell_origin == 'GCB':
            model_path = 'models/GCB/mfIHC/model-best.h5'
        elif cell_origin == 'non GCB':
            model_path = 'models/ABC/mfIHC/model-best.h5'
    elif image_type == 'pseudoIHC':
        if cell_origin == 'unknown':
            model_path = 'models/both/pseudoIHC/model-best.h5'
        elif cell_origin == 'GCB':
            model_path = 'models/GCB/pseudoIHC/model-best.h5'
        elif cell_origin == 'non GCB':
            model_path = 'models/ABC/pseudoIHC/model-best.h5'
    elif image_type == 'IHC':
        if cell_origin == 'unknown':
            model_path = 'models/both/IHC/model-best.h5'
        elif cell_origin == 'GCB':
            model_path = 'models/GCB/IHC/model-best.h5'
        elif cell_origin == 'non GCB':
            model_path = 'models/ABC/IHC/model-best.h5'
    
    try:
        model = load_model(model_path)
        return model
    except:
        st.error(f"Model not found at path: {model_path}. Please ensure the model file exists.")
        return None

def predict_patches(model, patches):
    # Original predict_patches function remains the same
    predictions = []
    for patch in patches:
        patch_array = np.array(patch.resize((256, 256))) 
        patch_array = np.expand_dims(patch_array, axis=0)
        prediction = model.predict(patch_array)[0][0]
        predictions.append(prediction)
    return predictions

def main():
    local_css()

    # Sidebar
    st.sidebar.image("CSi_logo.png", width=200)
    st.sidebar.title("About PhenoRelapseNet")
    st.sidebar.markdown("""
    ### Overview
    mDELRelapseNet is a deep learning-based tool for predicting relapse risk in Diffuse Large B-Cell Lymphoma (DLBCL) patients using multiplex immunohistochemistry images.
    
    ### Features
    - Multi-marker analysis (MYC, BCL2, BCL6)
    - Support for multiple image types
    - Cell of Origin (COO) consideration
    - Patch-based analysis
    
    ### How it Works
    1. Upload marker images
    2. Select image type and COO
    3. Process images for analysis
    4. Receive risk prediction
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### Technical Details
    - Model: Deep Convolutional Neural Network
    - Patch Size: 448x448 pixels
    - Resolution: 256x256 pixels
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("Created by Kanav - ADJ Lab CSI")
    st.sidebar.markdown("© 2025 ADJ Lab. All rights reserved.")

    # Main content
    st.title("mDELRelapseNet")
    st.subheader("Relapse Risk Prediction in DLBCL using Deep Learning", divider="gray")

    with st.expander("Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            image_type = st.selectbox(
                "Image Type",
                ["mfIHC", "pseudoIHC", "IHC"],
                help="Select the type of image analysis to perform"
            )
        with col2:
            cell_origin = st.selectbox(
                "Cell of Origin (Hans)",
                ["unknown" , "GCB", "non GCB",],
                help="Select the cell of origin classification"
            )

    # Image upload section
    st.header("Upload Images")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        myc_file = st.file_uploader("MYC Image", type=['png', 'jpg', 'jpeg'],
                                  help="Upload MYC marker image")
    with col2:
        bcl2_file = st.file_uploader("BCL2 Image", type=['png', 'jpg', 'jpeg'],
                                   help="Upload BCL2 marker image")
    with col3:
        bcl6_file = st.file_uploader("BCL6 Image", type=['png', 'jpg', 'jpeg'],
                                   help="Upload BCL6 marker image")

    if myc_file and bcl2_file and bcl6_file:
        with st.spinner("Processing images..."):
            myc_img = Image.open(myc_file)
            bcl2_img = Image.open(bcl2_file)
            bcl6_img = Image.open(bcl6_file)

            combined_img = combine_images(bcl2_img, myc_img, bcl6_img)

            st.header("🔄 Combined RGB Image")
            st.image(combined_img, caption="Combined RGB Image (BCL2-MYC-BCL6)", use_column_width=True)

            patch_size = 448
            black_threshold = 0.75

            process_button = st.button("🔍 Analyze Image", help="Click to start image analysis")
            
            if process_button:
                with st.spinner("Analyzing patches..."):
                    patches = create_patches(combined_img, patch_size)
                    st.success(f"Created {len(patches)} patches")

                    filtered_patches = [patch for patch in patches if is_valid_patch(patch, black_threshold)]
                    st.success(f"Valid patches after filtering: {len(filtered_patches)}")

                    model = load_appropriate_model(image_type, cell_origin)
                    if model:
                        predictions = predict_patches(model, filtered_patches)
                        
                        # Use list comprehension to separate predictions
                        above_threshold = [pred for pred in predictions if pred > 0.5]
                        below_threshold = [pred for pred in predictions if pred <= 0.5]
                        
                        # Determine which group is majority
                        if len(above_threshold) > len(below_threshold):
                            avg_prediction = sum(above_threshold) / len(above_threshold)  # Average of high-risk predictions
                            risk_level = "High Risk"
                        else:
                            avg_prediction = sum(below_threshold) / len(below_threshold)  # Average of low-risk predictions
                            risk_level = "Low Risk"
                        
                        # Determine which group is majority
                        if len(above_threshold) > len(below_threshold):
                            avg_prediction = np.mean(above_threshold)  # Average of only high-risk predictions
                            risk_level = "High Risk"
                        else:
                            avg_prediction = np.mean(below_threshold)  # Average of only low-risk predictions
                            risk_level = "Low Risk"
                        
                        # Results section
                        st.header("📊 Analysis Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            risk_level = "High Risk" if avg_prediction > 0.5 else "Low Risk"
                            risk_color = "#ff4b4b" if risk_level == "High Risk" else "#00cc96"
                            st.markdown(f"""
                                <div style='background-color: #080d07; padding: 20px; border-radius: 10px;'>
                                    <h3 style='color: {risk_color};'>Risk Assessment: {risk_level}</h3>
                                    <p>Confidence Score: {avg_prediction:.2%}</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("""
                                <div style='background-color: #080d07; padding: 20px; border-radius: 10px;'>
                                    <h3>Analysis Details</h3>
                                    <p>Based on patch-level analysis and molecular markers</p>
                                </div>
                            """, unsafe_allow_html=True)

                        # Display patches
                        st.subheader("🔍 Analyzed Patches")
                        cols = st.columns(4)
                        for idx, patch in enumerate(filtered_patches):
                            with cols[idx % 4]:
                                st.image(patch, caption=f"Patch {idx+1}\nScore: {predictions[idx]:.2f}", 
                                        use_column_width=True)

    # Hide Streamlit elements
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
