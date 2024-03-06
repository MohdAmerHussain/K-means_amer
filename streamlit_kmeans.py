# Import libraries
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from urllib.parse import quote
import pickle, joblib
from PIL import Image

# Load the saved model
model = pickle.load(open('clust_UNIV.pkl', 'rb'))
imp_enc_scale = joblib.load('imp_enc_scale')
outlier = joblib.load('winsor')


def predict(data, user, pw, db):
    engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))
    univ_df = data.drop(["UnivID", "Univ"], axis = 1)
    data1 = pd.DataFrame(imp_enc_scale.transform(univ_df), columns = imp_enc_scale.get_feature_names_out())
    data1[list(data1.iloc[:, -6:].columns)] = outlier.transform(data1[list(data1.iloc[:, -6:].columns)])
    prediction = pd.DataFrame(model.predict(data1), columns = ['cluster_id'])
    prediction = pd.concat([prediction, data], axis = 1)
    
    prediction.to_sql('university_pred_kmeans', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
    return prediction


def main():  
    st.set_page_config(layout="wide")



    # Define the file path of the image
    logo_path = r"c:\Users\mohda\Downloads\innologo.png"

    # Display the logo using PIL and Streamlit
    image = Image.open(logo_path)
    st.image(image, width=200, use_column_width=False)

    st.title("KMeans_Clustering")
    st.sidebar.title("KMeans_Clustering")

    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">KMeans_Clustering</h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    
    uploadedFile = st.sidebar.file_uploader("Choose a file", type = ['csv', 'xlsx'], accept_multiple_files = False, key = "fileUploader")
    if uploadedFile is not None :
        try:

            data = pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame(uploadedFile)
                
    else:
        st.sidebar.warning("You need to upload a csv or excel file.")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "")
    pw = st.sidebar.text_input("password", "", type="password")
    db = st.sidebar.text_input("database", "")
    
    result = ""
    
    if st.button("Predict"):
        result = predict(data, user, pw, db)
                                   
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap = True)
        st.table(result.style.background_gradient(cmap = cm))
                           
if __name__=='__main__':
    main()


