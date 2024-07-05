## 1. Project Overview <a class="anchor" id="project-description"></a>

Our team has been hired as data science consultants for a news outlet to create classification models using Python and deploy it as a web application with Streamlit. 
The aim is to provide you with a hands-on demonstration of applying machine learning techniques to natural language processing tasks.  This end-to-end project encompasses the entire workflow, including data loading, preprocessing, model training, evaluation, and final deployment. The primary stakeholders for the news classification project for the news outlet could include the editorial team, IT/tech support, management, readers, etc. These groups are interested in improved content categorization, operational efficiency, and enhanced user experience.

#### The methodology overview of our project consists of:
1. Project Overview
2. Importing Packages
3. Loading Data
4. Data Cleaning
5. Exploratory Data Analysis
6. Data Preprocessing 
7. Model training and evaluation
8. Model Improvement and MLFlow Integration 


## 2. Dataset <a class="anchor" id="dataset"></a>
The dataset is comprised of news articles that need to be classified into categories based on their content, including `Business`, `Technology`, `Sports`, `Education`, and `Entertainment`. You can find both the `train.csv` and `test.csv` datasets [here](https://github.com/ereshia/2401FTDS_Classification_Project/tree/main/Data/processed).

**Dataset Features:**
| **Column**                                                                                  | **Description**              
|---------------------------------------------------------------------------------------------|--------------------   
| Headlines   | 	The headline or title of the news article.
| Description | A brief summary or description of the news article.
| Content | The full text content of the news article.
| URL | The URL link to the original source of the news article.
| Category | The category or topic of the news article (e.g., business, education, entertainment, sports, technology).


## 3. Packages <a class="anchor" id="packages"></a>

To carry out all the objectives for this repo, the following necessary dependencies were loaded:
+ `Pandas 2.2.2` and `Numpy 1.26`
+ `Matplotlib 3.8.4`
+ `Seaborn 0.12.2`
+ `nltk 3.8.1`
+ `mlflow 2.4.1`
+ `re 2.2.1`


## 4. Streamlit<a class="anchor" id="streamlit"></a>

### What is Streamlit?

[Streamlit](https://www.streamlit.io/)  is a framework that acts as a web server with dynamic visuals, multiple responsive pages, and robust deployment of your models.

In its own words:
> Streamlit ... is the easiest way for data scientists and machine learning engineers to create beautiful, performant apps in only a few hours!  All in pure Python. All for free.

> It’s a simple and powerful app model that lets you build rich UIs incredibly quickly.

[Streamlit](https://www.streamlit.io/)  takes away much of the background work needed in order to get a platform which can deploy your models to clients and end users. Meaning that you get to focus on the important stuff (related to the data), and can largely ignore the rest. This will allow you to become a lot more productive.


## 5. Team Members<a class="anchor" id="team-members"></a>

| Name                                                                                        |  Email              
|---------------------------------------------------------------------------------------------|--------------------             
| [Clement Mphethi](https://github.com/HoOdpHarMxcisT)                                        | clementmphethi@gmail.com
| [Koena Mahladisa](https://github.com/koenaMahladisa)                                        | kmahladisa9@gmail.com
| [Naledi Mogale](https://github.com/Andriena)                                                | nalediandriena@gmail.com
| [Neo Radebe](https://github.com/umkhulubhungane)                                            | radebeneo17@gmail.com
| [Nolwazi Mndebele](https://github.com/NolwaziMND)                                           | mndebelenf@gmail.com
| [Tshepiso Mudau](https://github.com/tshepisoMudau)                                          | mudaureneilwe@gmail.com
