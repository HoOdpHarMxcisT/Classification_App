"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: ExploreAI Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib, os

# Data dependencies
import pandas as pd

# Function to load vectorizer and models
def load_resources():
    # Load vectorizer
    vectorizer_path = "tfidf_vectorizer.pkl"
    with open(vectorizer_path, 'rb') as f:
        vectorizer = joblib.load(f)
    
    # Load models
    mlr_model_path = "mlr_model.pkl"
    nb_model_path = "nb_model.pkl"
    gbc_model_path = "gbc_model.pkl"
    
    mlr_model = joblib.load(open(mlr_model_path, "rb"))
    nb_model = joblib.load(open(nb_model_path, "rb"))
    gbc_model = joblib.load(open(gbc_model_path, "rb"))
    
    return vectorizer, mlr_model, nb_model, gbc_model

# Load raw data
def load_raw_data():
    raw_data_path = "test.csv"
    raw_data = pd.read_csv(raw_data_path)
    return raw_data

# Main function where we will build the actual app
def main():
    """News Classifier App with Streamlit"""

    # Insert your logo above the title
    st.image("Datainsight_Logo.png", width=200)

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("DataInsight News Classifier")
    st.subheader("Analyzing news articles")

    # Define navigation menu options in the desired order
    menu = ["Home", "Overview", "About Us"]
    choice = st.sidebar.selectbox('Navigation', menu)

    # Building out the selected page
    if choice == 'Home':
        show_home_page()

    elif choice == 'Overview':
        show_overview_page()

    elif choice == 'About Us':
        show_about_us_page()

def show_home_page():
    st.info("**Prediction with ML Models**")
    # Creating a text box for user input
    news_text = st.text_area("Enter Text", "Type Here")

    st.info("**How to Use the News Classifier App**")
    st.markdown("""
    ### How to Use the News Classifier App

    1. **Enter Your Text:**
       - In the text area provided above, type or paste the news article you want to classify.

    2. **Select Model:**
       - Choose a model from the dropdown below to classify the text.

    3. **Classify the Text:**
       - Click the "Classify" button. The app will use the selected model to analyze and classify the text.

    4. **View the Result:**
       - The classification result will be displayed on the screen, indicating the predicted category of the news article.
    """)

    # Model selection dropdown
    model_choice = st.selectbox("Choose Model", ("Logistic Regression", "Naive Bayes", "Gradient Boosting"))

    if st.button("Classify"):
        # Load resources
        vectorizer, mlr_model, nb_model, gbc_model = load_resources()

        # Transforming user input with vectorizer
        vect_text = vectorizer.transform([news_text]).toarray()

        # Make predictions based on selected model
        if model_choice == "Logistic Regression":
            prediction = mlr_model.predict(vect_text)
        elif model_choice == "Naive Bayes":
            prediction = nb_model.predict(vect_text)
        elif model_choice == "Gradient Boosting":
            prediction = gbc_model.predict(vect_text)

        # Display predicted category
        st.success("Predicted Category: {}".format(prediction[0]))  # Assuming prediction is a single value or array

def show_overview_page():
    st.info("**Proudly brought to you by DataInsight Solutions!**")
    st.markdown("This app allows you to classify news articles using machine learning models. "
                "You can navigate to the Home page to classify new articles or visit the About Us page to learn more about DataInsight Solutions.")

    # Insert an image on the homepage
    st.image("Homepage.jpg", caption="The power of predictive analysis is within your reach", use_column_width=True)

    # Insert case study content
    st.subheader("Enhancing News Classification")
    st.markdown("""
    In today’s digital age, efficiently managing the vast amount of news content is a significant challenge for news outlets. Our team at DataInsight Solutions has been brought on board as data science consultants to develop a sophisticated news classification system. We will use machine learning and natural language processing (NLP) to improve content categorization and enhance the reader experience.

    We will build an end-to-end system that includes data loading, preprocessing, model training, evaluation and deployment through a user-friendly Streamlit interface. This will ensure accurate classification of news articles, optimizing content management for the outlet and providing a more personalized experience for readers.

    The key stakeholders who will benefit from our solution are:

    - **Editorial Team:** Simplified workflows and better article organization.
    - **IT/Tech Support:** Easy integration and deployment of advanced models.
    - **Management:** Increased operational efficiency and valuable strategic insights.
    - **Readers:** More personalized and engaging news content.
    """)

    st.markdown("**Grow with us.** [Click here to find out more](#About_Us)")

    # Insert video on the homepage
    st.subheader("Reporting live: Political Scandal Unveiled, High-level Corruption Exposed")
    st.video("Breaking News Video.mp4")

def show_about_us_page():
    st.info("**About Us**")
    st.markdown("""
    ### Founding Story of DataInsight Solutions

    Founded in 2015 in South Africa, DataInsight Solutions emerged from the vision of a team of passionate data scientists and industry experts. Recognizing the transformative potential of data, our journey began with a commitment to bridging the gap between data and decision-making for businesses.

    From our modest beginnings, we have grown into a trusted partner, employing and training over 200 data practitioners. Our foundation is built on innovation, integrity, and a client-centric approach, driving us to continually explore new frontiers in data science, machine learning, and artificial intelligence.

    ### Our Purpose

    At DataInsight Solutions, our purpose is to empower businesses with actionable insights derived from data. We believe in harnessing the power of data science and analytics to solve complex challenges, drive innovation, and enable informed decision-making across industries.

    ### Our Vision

    Our vision at DataInsight Solutions is to be a leading provider of data-driven solutions globally. We strive to be recognized for our expertise in transforming data into valuable insights that drive business growth, enhance operational efficiency, and deliver exceptional value to our clients.
    """)

    # Meet Our Team section
    st.markdown("### Meet Our Team")
    st.markdown("""
    - **Clement Mphethi** - Lead Data Scientist
    - **Nolwazi Mndebele** - Project Manager
    - **Tshepiso Mudau** - Github Manager
    - **Neo Radebe** - Data Scientist
    - **Naledi Mogafe Mogale** - Data Scientist
    - **Koena Mcdonald Mahladisa** - Data Scientist
    """)
    
    # Contact Us section
    st.markdown("### Contact Us:")
    st.markdown("For inquiries, please contact us at [info@datainsight.com](mailto:info@datainsight.com).")

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()