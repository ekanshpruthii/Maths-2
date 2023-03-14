import streamlit as st
import pickle
import numpy as np

import asyncio
import subprocess

# create a function that takes in input features and returns a prediction
async def predict(x):
    # load the saved model
    with open('banknote_model.pkl', 'rb') as file:
        model = pickle.load(file)
        
    # make a prediction using the loaded model
    y_pred = model.predict(x)
    return y_pred

# create a Streamlit app
def main():
    st.title('Mini Project : Bank Note Authentication Model')
    st.write('This is a neural network model for bank note authentication.')
    
    # get input from user
    variance = st.text_input('Variance', '')
    skewness = st.text_input('Skewness', '')
    curtosis = st.text_input('Curtosis', '')
    entropy = st.text_input('Entropy', '')
        
    # convert the input to a numpy array
    x_input = np.array([[variance, skewness, curtosis, entropy]])


    if st.button("Predict"):
        # use asyncio to run the predict_bank_notes function asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        y_pred = loop.run_until_complete(predict(x_input))
        loop.close()

        # display the prediction result
        if y_pred is not None:
            st.write("Prediction Result:", int(y_pred[0]))

            if int(y_pred[0]) == 0:
                st.write("Not Authentic")
            elif int(y_pred[0]) == 1:
                st.write("Authentic")


if __name__ == '__main__':
    main()