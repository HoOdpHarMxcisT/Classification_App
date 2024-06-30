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

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Home", "Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Home" page
    if selection == "Home":
        st.info("Welcome to DataInsight Solutions!")
        st.markdown("This app allows you to classify news articles using machine learning models. "
                    "You can navigate to the Prediction page to classify new articles or visit the Information page to learn more about the app.")
        
        # Insert an image on the homepage
        st.image("Homepage.jpg", caption="The power of predictive analysis is within your reach", use_column_width=True)


        
        # Insert case study content
        st.subheader("Enhancing News Classification")
        st.markdown("""
        In today’s digital age, efficiently managing the vast amount of news content is a significant challenge for news outlets. Our team at DataInsight Solutions has been brought on board as data science consultants to develop a sophisticated news classification system. This project will use machine learning and natural language processing (NLP) to improve content categorization and enhance the reader experience.

        Our purpose is to ensure accurate classification of news articles, optimizing content management for the outlet and providing a more personalized experience for readers.

        The key stakeholders who will benefit from our solution are:

        - **Editorial Team:** Simplified workflows and better article organization.
        - **IT/Tech Support:** Easy integration and deployment of advanced models.
        - **Management:** Increased operational efficiency and valuable strategic insights.
        - **Readers:** More personalized and engaging news content.

        DataInsight Solutions aims to showcase the practical application of data science in solving real-world problems. Our project will emphasize the importance of thorough data preprocessing, selecting appropriate models, and rigorous performance evaluation. Deploying our solution with Streamlit ensures accessibility and usability for non-technical stakeholders.
        """)

        # Insert video on the homepage
        st.markdown("### In today's news...")
        st.subheader("Reporting live: Political Scandal Unveiled, High-level Corruption Exposed")
        st.video("Breaking News Video.mp4")

        st.markdown("### Services We Offer:")
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

    # Building out the "Information" page
    if selection == "Information":
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

        ### Benefits of Using the App

        - **Quick and Accurate Classification:**
          Our app leverages advanced machine learning algorithms to provide fast and accurate classification of news articles.

        - **User-Friendly Interface:**
          The app is designed to be intuitive and easy to use, even for those without a technical background.

        - **Versatile Applications:**
          Whether you're a journalist, researcher, or just someone interested in categorizing news, this app can be a valuable tool.

        ### Contact Us

        For more information or support, please contact us at [support@datainsight.com](mailto:support@datainsight.com).
        """)

    # Building out the prediction page
    if selection == "Prediction":
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

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    main()

