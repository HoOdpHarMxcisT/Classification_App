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

# Vectorizer
# news_vectorizer = open("streamlit/tfidfvect.pkl", "rb")
# test_cv = joblib.load(news_vectorizer)  # loading your vectorizer from the pkl file

# Load your raw data
# raw = pd.read_csv("streamlit/train.csv")

# The main function where we will build the actual app
def main():
    """News Classifier App with Streamlit"""

    # Insert your logo above the title
    st.image("Datainsight_Logo.png", width=200)

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("DataInsight News Classifier")
    st.subheader("Analyzing news articles")

    # Define navigation menu options in the desired order
    menu = ["Home", "Instructions", "Prediction", "About Us"]
    choice = st.sidebar.selectbox('Navigation', menu)

    # Building out the selected page
    if choice == 'Home':
        show_homepage()

    elif choice == 'Instructions':
        show_instructions_page()

    elif choice == 'Prediction':
        show_prediction_page()

    elif choice == 'About Us':
        show_about_us_page()


def show_homepage():
    st.info("Welcome to DataInsight Solutions!")
    st.markdown("This app allows you to classify news articles using machine learning models. "
                "You can navigate to the Prediction page to classify new articles or visit the Instructions page to learn more about the app.")

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

    st.markdown("**Grow your business with us.** [Click here to find out more](#About_Us)")

    # Insert video on the homepage
    st.subheader("Reporting live: Political Scandal Unveiled, High-level Corruption Exposed")
    st.video("Breaking News Video.mp4")


def show_instructions_page():
    st.info("How the App Works")
    st.markdown("""
    ### How to Use the News Classifier App

    1. **Navigate to the Prediction Page:**
       - Use the sidebar to select the "Prediction" option.

    2. **Enter Your Text:**
       - In the text area provided, type or paste the news article you want to classify.

    3. **Classify the Text:**
       - Click the "Classify" button. The app will use a pre-trained machine learning model to analyze and classify the text.

    4. **View the Result:**
       - The classification result will be displayed on the screen, indicating the category of the news article.

    ### Behind the Scenes

    - **Data Preprocessing:**
      The text you input is preprocessed to remove any noise and convert it into a format suitable for the machine learning model.

    - **Vectorization:**
      The cleaned text is transformed into numerical vectors using a technique called TF-IDF (Term Frequency-Inverse Document Frequency).

    - **Prediction:**
      The vectorized text is fed into a machine learning model (e.g., Logistic Regression) to predict the category of the news article.

    - **Output:**
      The predicted category is displayed to the user in a human-readable format.

    ### Why should you use our News Classifier App?.

    - **Quick and Accurate Classification:**
      Our app leverages advanced machine learning algorithms to provide fast and accurate classification of news articles.

    - **User-Friendly Interface:**
      The app is designed to be intuitive and easy to use, even for those without a technical background.

    - **Versatile Applications:**
      Whether you're a journalist, researcher, or just someone interested in categorizing news, this app can be a valuable tool.

    ### Contact Us

    For more information or support, please contact us at [support@datainsight.com](mailto:support@datainsight.com).
    """)


def show_prediction_page():
    st.info("Prediction with ML Models")
    # Creating a text box for user input
    news_text = st.text_area("Enter Text", "Type Here")

    if st.button("Classify"):
        # Transforming user input with vectorizer
        vect_text = test_cv.transform([news_text]).toarray()
        # Load your .pkl file with the model of your choice + make predictions
        # Try loading in multiple models to give the user a choice
        predictor = joblib.load(open(os.path.join("streamlit/Logistic_regression.pkl"), "rb"))
        prediction = predictor.predict(vect_text)

        # When model has successfully run, will print prediction
        # You can use a dictionary or similar structure to make this output
        # more human interpretable.
        st.success("Text Categorized as: {}".format(prediction))


def show_about_us_page():
    st.info("About Us")
    st.markdown("""
    ### Founding Story of DataInsight Solutions

    Founded in 2015 in South Africa, DataInsight Solutions emerged from the vision of a team of passionate data scientists and industry experts. Recognizing the transformative potential of data, our journey began with a commitment to bridging the gap between data and decision-making for businesses.

    From our modest beginnings, we have grown into a trusted partner, employing and training over 200 data practitioners. Our foundation is built on innovation, integrity, and a client-centric approach, driving us to continually explore new frontiers in data science, machine learning, and artificial intelligence.

    ### Our Purpose

    At DataInsight Solutions, our purpose is to empower businesses with actionable insights derived from data. We believe in harnessing the power of data science and analytics to solve complex challenges, drive innovation, and enable informed decision-making across industries.

    ### Our Vision

    Our vision at DataInsight Solutions is to be a leading provider of data-driven solutions globally. We strive to be recognized for our expertise in transforming data into valuable insights that drive business growth, enhance operational efficiency, and deliver exceptional value to our clients.
    """)

    # Move services information here
    st.markdown("### Our Services:")
    st.markdown("""
    - **Data Strategy and Consulting**
    - **Machine Learning and Artificial Intelligence**
    - **Natural Language Processing (NLP)**
    - **Data Engineering**
    - **Data Visualization and Reporting**
    - **Cloud Solutions**
    - **Advanced Analytics**
    - **Training and Workshops**
    """)


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()