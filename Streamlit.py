import numpy as np
import streamlit as st
import pickle as pk
with open('Crop recommendation.pkl','rb') as file:
    loaded_model1=pk.load(file)




def crop_recommend(input_data):
    i_data=np.asarray(input_data)
    final_data=i_data.reshape(1,-1)
    result=loaded_model1.predict(final_data)
    return str(result[0])

def main():
    st.title('Crop Recommendation')
    temperature=st.text_input('Temperature of the field region')
    humidity=st.text_input('Humidity of the field region')
    ph=st.text_input('pH of the Crop')
    rainfall=st.text_input('Rainfall near the field region')
    input_data=[temperature,humidity,ph,rainfall]
    recommend=''
    if st.button('Click for crop name'):
        recommend=crop_recommend(input_data)
    st.success(recommend)

if __name__=='__main__':
    main()

