import streamlit as st
from telco import * 
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from catboost.utils import get_roc_curve
from sklearn.metrics import confusion_matrix, classification_report, auc

import plotly.figure_factory as ff
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import os 
from pathlib import Path


st.set_page_config(page_title='Telco Churn Prediction', layout='wide')

st.write("""
# Telco Churn Prediction
         
A web application to predict and analyze Telco Churn dataset.
Powered by `streamlit`.
         
""")
st.divider()

sidebar = ('Exploratory Data Analysis', 'Machine Learning Model Computation', 'Churn Probability Prediction')
with st.sidebar:
    st.markdown('##### Author : Rasyid Sulaeman')
    st.markdown('##### Check out the repo [here](https://github.com/rasyidsulaeman/Telco_Churn_Prediction)')
    st.header('Select Data Processess')
    radio = st.radio('Modes', options=sidebar)

path = 'dataset/churn.csv'
telco = TelcoEDA(path)
df = telco.data_loaded()

df['SeniorCitizen'] = df['SeniorCitizen'].map(pd.Series({1 : 'Yes', 0 : 'No'})).astype('object')

def general_info(df):
    st.write('#### General Info')

    total_missing = telco.total_missing()
    duplicated = telco.total_duplicated()
    
    cols = st.columns(5)
    cols[0].metric('Total columns', len(df.columns))
    cols[1].metric('Total entries', len(df))
    cols[2].metric('Missing Value Percentage', str(total_missing) + '%')
    cols[3].metric('Total Duplicate Data ', duplicated[0])
    cols[4].metric('Duplicated Percentage', str(duplicated[1]) + '%')

def visualization(df, plot):
    st.write('#### Visualization')

    st.write('##### Bivariate Analysis - Category Feature')
    category_columns = df.select_dtypes('object').columns.tolist()[1:-1]
    cat_choose = st.selectbox('Select feature to display', category_columns)
    fig = plot.bivariate_category(cat_choose)
    st.plotly_chart(fig, use_container_width=True)

    st.write('##### Bivariate Analysis - Numeric Feature')
    numeric_columns = df.select_dtypes('float64').columns.tolist()
    num_choose = st.selectbox('Select feature to display', numeric_columns)
    fig = plot.bivariate_numeric(num_choose)
    st.plotly_chart(fig, use_container_width=True)

 # save the model
model_dir = str(Path().absolute()) + '/model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, "catboost_model.cbm")

if radio == sidebar[0]:
    st.write('#### Telco Churn Dataset')
    st.dataframe(df) 

    # display general info of the dataset
    general_info(df)

    # show missing value for each feature
    missing = telco.missing_value(df)
    st.write("""
    #### Missing Value
            
    Compute missing value in our Telco Churn dataset
    """)
    st.dataframe(missing.astype(str))

    # visualize the feature
    telco_plot = TelcoPlot(df=df)
    visualization(df, telco_plot)

elif radio == sidebar[1]:
    
    st.write('## Machine Learning - Categorical Boosting Model')

    # remove na
    df = telco.remove(df, bool=True)

    # data cleaning
    df['Churn'] = df['Churn'].map(pd.Series({'Yes' : 1, 'No' : 0})).astype('int64')
    
    # get feature and target data
    feature = df.drop(['customerID', 'Churn'], axis=1)
    target = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)

    st.markdown('A model is being built to predict the following **Y** variable:')
    st.info(target.name)

    st.write('#### Tune the hyperparameter')

    parameter_columns = st.columns(2)

    with parameter_columns[0]:
        iterations = st.slider('Number of iterations', 0, 500, 50)
        learning_rate = st.slider('Learning rate', 0.01, 1.0, step=0.02)
        max_depth = st.slider('The maximum depth of the tree', 1, 50, 2)
    with parameter_columns[1]:
        loss_function = st.select_slider('Loss function', options=['Logloss', 'CrossEntropy'])
        l2_leaf_reg = st.slider('Leaf values to L2 regularization', 0, 200, 20)
        border_count = st.select_slider('The number of splits for numerical features', options=[1, 4, 16, 32, 64, 128])

    categorical = feature.select_dtypes('object').columns.tolist()
    cboost = CatBoostClassifier(iterations = iterations,
                                learning_rate = learning_rate,
                                depth = max_depth,
                                l2_leaf_reg = l2_leaf_reg,
                                loss_function=loss_function,
                                border_count = border_count,
                                verbose=False,
                                random_state=42)
    
    cboost.fit(X_train, y_train, cat_features = categorical, eval_set=(X_test, y_test))

    y_pred = cboost.predict(X_test)

    st.write('#### Confusion Matrix')
    st.markdown('Performance measurement for classification model')
    cm = confusion_matrix(y_test, y_pred)
    cm_text = [[str(y) for y in x] for x in cm]
    labels = ['No Churn', 'Churn']
    fig = ff.create_annotated_heatmap(cm, x=labels, y=labels, annotation_text=cm_text, colorscale='Blues')
    st.plotly_chart(fig, use_container_width=True)

    st.write('#### Classification Report')
    report = classification_report(y_test, y_pred, output_dict=True)

    st.markdown('Accuracy of our classification model')
    st.info(report['accuracy'])

    metrics = pd.DataFrame([report['0'], report['1']], index=['No Churn', 'Churn'])

    st.markdown('Overall classification report for each class including **Precision**, **Recall**, **F1 Score**, **Support**')
    st.dataframe(metrics, use_container_width=True)

    st.write('#### Save your model')
    st.warning('You can saved the already tuned model: Click saved model')
    save = st.button('Save the model')
    if save:
        cboost.save_model(model_path)

    st.write('#### ROC AUC Curve')
    st.markdown('ROC AUC represents how much our model is capable of distinguishing between classes')
    dataset = Pool(X_test, y_test, cat_features=categorical)
    fpr, tpr, _ = get_roc_curve(cboost, dataset, plot=False)
    auc_plot = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                             name = f'ROC AUC Score: {round(auc_plot,3)}'))

    fig.update_layout(title_text=f'ROC AUC Curve of Telco Churn Prediction',
                        xaxis_title = 'False Positive Rate',
                        yaxis_title = 'True Positive Rate', showlegend=True)
    
    st.plotly_chart(fig, use_container_width=True)

    # Explaining the model's predictions using SHAP values
    explainer = shap.Explainer(cboost, feature_perturbation="tree_path_dependent")
    shap_values = explainer(X_test)

    st.write('#### Feature Importance') 
    st.markdown('Visualized feature contribution using `shap` package')
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    shap.plots.bar(shap_values, max_display=15, show=False)
    st.pyplot(fig)

elif radio == sidebar[2]:
    
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    st.write('#### Telco Churn Prediction')
    st.write('Adjust following feature to your preferences to predict whether you will **Churn** or **Not Churn**')
    gender = st.selectbox('Gender', ('Male', 'Female'))
    seniorcitizen = st.selectbox('Senior Citizen', ('Yes', 'No'))
    partner = st.selectbox('Partner', ('Yes', 'No'))
    dependents = st.selectbox('Dependents', ('Yes', 'No'))
    tenure = st.number_input('Tenure', min_value=0, max_value= int(df['Tenure'].max()), step=1)
    contract = st.selectbox('Contract', ('One year', 'Two year', 'Month-to-month'))
    paperlessbilling = st.selectbox('Paperless Billing', ('Yes', 'No'))
    paymentmethod = st.selectbox('Payment Method', ('Electronic check', 'Mailed check', 
                                                    'Bank transfer (automatic)', 'Credit card (automatic)'))
    monthlycharges = st.number_input("Monthly Charges", min_value=0.0, max_value=df['MonthlyCharges'].max(), step=0.01)
    totalcharges = st.number_input("Total Charges", min_value=0.0, max_value=df['TotalCharges'].max(), step=0.01)
        
    confirm = st.button("Confirm")

    if confirm:
        new_customer_data = pd.DataFrame({'Gender' : [gender],
                                          'SeniorCitizen' : [seniorcitizen],
                                          'Partner' : [partner],
                                          'Dependents' : [dependents],
                                          'Tenure' : [tenure],
                                          'Contract' : [contract],
                                          'PaperlessBilling' : [paperlessbilling],
                                          'PaymentMethod' : [paymentmethod],
                                          'MonthlyCharges' : [monthlycharges],
                                          'TotalCharges' : [totalcharges]})
        
        st.markdown('This is your selected parameters')
        st.dataframe(new_customer_data, use_container_width=True)

        predict = model.predict(new_customer_data)
        prob = model.predict_proba(new_customer_data)[:,1].item()

        if predict[0] == 0:
            pred = 'Not Churn'
        else:
            pred = 'Churn'

        st.info(f"""
        #### Churn Prediction
                
        Most likely you will **{pred}** because Your churning probability is **{round(prob*100,2)} %**
                """)