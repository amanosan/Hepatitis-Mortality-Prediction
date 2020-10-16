from user_data_db import *
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import hashlib
from lime.lime_tabular import LimeTabularExplainer

sns.set_palette(palette='viridis')

# function to plot relation between two features:


def plot_relation(feature1, feature2, df):
    g = sns.JointGrid(x=feature1, y=feature2, data=df)
    g.plot_joint(sns.scatterplot, alpha=0.7)
    g.plot_marginals(sns.distplot, kde=True)
    st.pyplot()

# function to convert the password to hash.


def passowrd_hash(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


# function to verify hashes and password:
def verify_hash(password, hash_text):
    if passowrd_hash(password) == hash_text:
        return hash_text
    return False


# the features we have used in our ML models:
features = ['protime', 'sgot', 'bilirubin', 'age', 'alk_phosphate', 'albumin', 'spiders', 'histology', 'fatigue',
            'ascites', 'varices', 'sex', 'antivirals', 'steroid']

gender_dict = {'Male': 1, 'Female': 2}
categorical_dict = {'No': 1, 'Yes': 2}


def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

# function to get user inputs:


def get_user_input():
    age = st.number_input(label="Enter Age:", value=0, step=1)

    # sex
    sex_input = st.radio(label="Select Gender: ", options=['Male', 'Female'])
    sex = get_value(sex_input, gender_dict)

    # steroid
    steroid_input = st.radio(
        label="Are you on Steroids?", options=['Yes', 'No'])
    steroid = get_value(steroid_input, categorical_dict)

    # antivirals
    antivirals_input = st.radio(
        label="Do you take Antivrirals?", options=['Yes', 'No'])
    antivirals = get_value(antivirals_input, categorical_dict)

    # spiders
    spiders_input = st.radio(
        label="Presence of Spider Naevus?", options=['Yes', 'No'])
    spiders = get_value(spiders_input, categorical_dict)

    # ascites
    ascites_input = st.radio(label="Ascites?", options=['Yes', 'No'])
    ascites = get_value(ascites_input, categorical_dict)

    # varices
    varices_input = st.radio(
        label="Presence of Varices?", options=['Yes', 'No'])
    varices = get_value(varices_input, categorical_dict)

    # histology
    histology_input = st.radio(label="Histology?", options=['Yes', 'No'])
    histology = get_value(histology_input, categorical_dict)

    # fatigue
    fatigue_input = st.radio(label='Are you Fatigued?', options=['Yes', 'No'])
    fatigue = get_value(fatigue_input, categorical_dict)

    # bilirubin
    bilirubin = st.slider(label="Bilirubin Content: ",
                          value=0.0, step=0.1, max_value=8.0)

    # sgot
    sgot = st.number_input(label="SGOT:", value=0.0, step=0.1)

    # protime
    protime = st.number_input(label="Protime:", value=0.0, step=0.1)

    # alk_phosphate
    alk_phosphate = st.number_input(
        label="Alkaline Phosphate:", value=0.0, step=0.1)

    # albumin
    albumin = st.slider(label="Albumin:", value=0.0, max_value=8.0, step=0.1)

    return [protime, sgot, bilirubin, age, alk_phosphate, albumin, spiders, histology, fatigue,
            ascites, varices, sex, antivirals, steroid]


# function to get select the Machine Learning model:
def get_model():
    st.subheader(
        "Please select a Machine Learning Model to predict the result:")
    model = st.selectbox(label="", options=[
                         'Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest'])

    if model == 'Logistic Regression':
        clf = pickle.load(open(r'models/logreg_model.pkl', 'rb'))

    elif model == 'KNN':
        clf = pickle.load(open(r'models/knn_model.pkl', 'rb'))

    elif model == 'Decision Tree':
        clf = pickle.load(open(r'models/dt_model.pkl', 'rb'))

    elif model == 'Random Forest':
        clf = pickle.load(open(r'models/rf_model.pkl', 'rb'))

    return clf


# the main function
def app():
    """ Mortality Prediction App """
    st.title("Disease Mortality Prediction App")

    menu = ['Home', 'Login', 'Sign Up']
    submenu = ['Plot', 'Prediction']

    choice = st.sidebar.selectbox(label='Menu', options=menu)

    if choice == 'Home':
        st.subheader("Home")

        st.write(
            """
            ### What is Hepatitis B?
            A serious liver infection caused by the hepatitis B virus that's easily preventable by a vaccine.\n
            This disease is most commonly spread by exposure to infected bodily fluids.
            Symptoms are variable and include yellowing of the eyes, abdominal pain and dark urine. \
            Some people, particularly children, don't experience any symptoms. \n
            In chronic cases, liver failure, cancer or scarring can occur.\
            The condition often clears up on its own. Chronic cases require medication and possibly a liver\
            transplant.

            ### Symptoms: 
            - Fever, fatigue, muscle or joint pain,
            - Loss of appetitie,
            - Mild nausea and vomitting,
            - Stomach pain,
            - Bloated or swollen stomach,
            - Yellow skin/eyes (called Jaundice)

            ### Treatment:
            The treatment depends on the severity.\n
            The condition often clears up on its own. Chronic cases require medication and possibly a liver\
                transplant. 
            """
        )

    elif choice == 'Login':
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type='password')

        if st.sidebar.checkbox("Login"):
            create_usertable()
            hashed_pswd = passowrd_hash(password)
            result = login_user(username, verify_hash(password, hashed_pswd))
            # if password == '12345':
            if result:
                st.success(f"Welcome, {username}.")
                task = st.selectbox(
                    label="Choose a Task to perform:", options=submenu)

                # plotting data:
                if task == 'Plot':
                    st.header("Data Visualization")

                    # plotting the class plot
                    st.subheader("Looking at the Training Data")
                    df = pd.read_csv(r"data/clean_hepatitis_dataset.csv")
                    st.dataframe(df)

                    plt.figure()
                    sns.countplot(x='class', data=df)
                    plt.xticks(ticks=[0, 1], labels=['Die', 'Live'])
                    plt.ylabel('Count')
                    plt.xlabel('Class')
                    st.pyplot()

                    # plotting frequency distribution
                    st.subheader('Frequency Distribution by Age')
                    freq_df = pd.read_csv(
                        r'data/freq_distribution_hepatitis_dataset.csv')
                    st.dataframe(freq_df[1:])

                    plt.figure()
                    sns.barplot(x='age', y='count', data=freq_df)
                    plt.xlabel('Distribution')
                    plt.ylabel('Count')
                    st.pyplot()

                    features = df.columns.to_list()
                    # plotting relation between any two features:

                    if st.checkbox("Relationship Plot"):
                        x_axis = st.selectbox(
                            label="Select X-axis: ", options=features)
                        y_axis = st.selectbox(
                            label="Select Y-axis: ", options=[x for x in features if x is not x_axis])
                        plot_relation(x_axis, y_axis, df)

                    # plotting area chart
                    if st.checkbox("Area Chart"):
                        selected_features = st.multiselect(
                            "Choose Features:", options=features)
                        new_df = df[selected_features]
                        st.area_chart(data=new_df)

                # predictions:
                elif task == "Prediction":
                    st.subheader("Predictive Analysis")
                    user_input = get_user_input()
                    clf = get_model()
                    if st.button('Predict'):
                        prediction = clf.predict([user_input])
                        prob_live = clf.predict_proba([user_input])[0][1]
                        prob_die = clf.predict_proba([user_input])[0][0]
                        if prediction == 1:
                            st.warning(
                                f"Patient has {prob_die:.2f} chances of dying.")
                        elif prediction == 2:
                            st.success(
                                f"Patient has {prob_live:.2f} chances of living")

                    # Interpretation of our model.
                    if st.checkbox("Interpret the Model result."):

                        df = pd.read_csv(r'data/clean_hepatitis_dataset.csv')
                        feature_names = ['protime', 'sgot', 'bilirubin', 'age', 'alk_phosphate', 'albumin', 'spiders', 'histology', 'fatigue',
                                         'ascites', 'varices', 'sex', 'antivirals', 'steroid']
                        x = df[feature_names]
                        class_names = ['Die(1)', 'Live(2)']

                        explainer = LimeTabularExplainer(x.values, feature_names=feature_names,
                                                         class_names=class_names, discretize_continuous=True)

                        exp = explainer.explain_instance(np.array(user_input), clf.predict_proba,
                                                         num_features=14, top_labels=1)

                        score_list = exp.as_list()
                        label_limits = [i[0] for i in score_list]
                        label_scores = [i[1] for i in score_list]

                        plt.barh(label_limits, label_scores)
                        st.pyplot()

                        plt.figure(figsize=(20, 10))
                        fig = exp.as_pyplot_figure()
                        st.pyplot()

            else:
                st.warning("Incorrect username/password.")

    elif choice == 'Sign Up':
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type='password')
        confirm_password = st.text_input("Confir Passoword", type='password')

        if confirm_password == new_password:
            st.success("Password Confirmed.")
        else:
            st.warning("Passwords do not match.")

        if st.button("Submit"):
            create_usertable()
            hashed_new_password = passowrd_hash(new_password)
            add_userdata(username=new_username, password=hashed_new_password)
            st.success("You have successfully created a new account.")
            st.info("Login to get started.")


if __name__ == "__main__":
    app()
