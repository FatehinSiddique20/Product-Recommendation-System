import streamlit as st 
import streamlit.components.v1 as stc 


import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline
plt.style.use("ggplot")

import sklearn
import neattext.functions as nfx
from sklearn.decomposition import TruncatedSVD
import urllib.request
from PIL import Image

def prod_name(data,user_id):
    name = (data.loc[data['user_id'] == user_id]).head(1)
    #name = name[1]
    x = name['name']
    x = str(x)
    x = x.split()
    num = len(x)-4
    x = x[1:num]
    x = ' '.join(x)
    return x

def load_data(data):
    df = pd.read_csv(data)
    return df 

def vec_text_to_tfidf_mat(data):
    tfidf_vect = TfidfVectorizer()
    tf_mat = tfidf_vect.fit_transform(data)
    cosine_sim_tf_mat = cosine_similarity(tf_mat)
    return cosine_sim_tf_mat

@st.cache
def get_recommendation_cat(title,cosine_sim_tf_mat,df,num_of_rec):
    # indices of the product
    product_indices = pd.Series(df.index,index=df['name']).drop_duplicates()
    # Index of product
    idx = product_indices[title]
    sim_scores =list(enumerate(cosine_sim_tf_mat[idx]))
    sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)
    selected_product_indices = [i[0] for i in sim_scores[1:]]
    selected_product_scores = [i[1] for i in sim_scores[1:]]
    # Get the dataframe & title
    result_df = df.loc[selected_product_indices]
    result_df['similarity_score'] = selected_product_scores
    #result_df['similarity_score'] = scaler.fit_transform(result_df['similarity_score'])
    final_recommended_product = result_df[['imageURLs','name','brand','weight','rating','similarity_score']]
    return final_recommended_product.head(num_of_rec)

def pop_product (data, num):
    popular_products = data[['imageURLs','name','brand','weight','rating']]
    most_popular = popular_products.sort_values('rating', ascending=False)
    #most_popular = most_popular.reset_index()
    return most_popular.head(num)

# Search For Product 
@st.cache
def search_term_if_not_found(term,df):
    result_df = df[df['name'].str.contains(term)]
    return result_df[['imageURLs','name','brand','weight','rating']]

RESULT_TEMP2 = """
<div style="width:90%;height:100%;margin:1px;padding:3px;position:relative;border-radius:0px;border-bottom-right-radius: 10px;
box-shadow:0 0 15px 5px #ccc; background-color: #C0C0C0;
  border-left: 5px solid #6c6c6c;">
<img src="{}" style="width:100%">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">Brand:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Weight:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Rating:</span>{}</p>
</div>
"""    

RESULT_TEMP3 = """
<div style="width:90%;height:100%;margin:1px;padding:3px;position:relative;border-radius:0px;border-bottom-right-radius: 10px;
box-shadow:0 0 15px 5px #ccc; background-color: #C0C0C0;
  border-left: 5px solid #6c6c6c;">
<img src="{}" style="width:100%">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">Brand:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Weight:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Rating:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Similarity Score:</span>{}</p>
</div>
"""    


def main():
    
    
        

    
    image = Image.open("D:/Elec_rec/logo.png")
    st.sidebar.image(image,width=200)
        
    st.title("E-Commerce Product Recommendation Demo")

    menu = ["Home","Recommend","User-based Recommend","About"]
    choice = st.sidebar.selectbox("Menu",menu)

    df1 = load_data("D:/Elec_rec/elec_product_data.csv")
    df1['user_id'] = df1['id']
    df1['product_id'] = df1['asins']
    df1['clean_name'] = df1['name'].apply(nfx.remove_stopwords)

    data = df1[['user_id','product_id','rating','name','brand','weight','categories','primaryCategories','imageURLs','clean_name']]
    x = data.groupby(['product_id','user_id','name','clean_name','brand','weight','categories','primaryCategories','imageURLs'])['rating'].mean()
    group_data = pd.DataFrame(x)
    group_data = group_data.reset_index()
    #df = group_data
    df = load_data("D:/Elec_rec/group_data.csv")


    if choice == "Home":
        st.subheader("Home")
        st.dataframe(df.head(10))
        st.subheader("Trending Now")
        y = st.sidebar.slider('Number of Popular Product',min_value = 5, max_value=10,step=1,value=5)
        num = st.write("Total Recommendations:",y)
        num=y
        top_product = pop_product (df,num)
        for row in top_product.iterrows():
            rec_imageurl = row[1][0]
            rec_title = row[1][1]
            rec_brand = row[1][2]
            rec_weight = row[1][3]
            rec_rating = row[1][4]
            
            
            
            stc.html(RESULT_TEMP2.format(rec_imageurl,rec_title,rec_brand,rec_weight,rec_rating),height=950)
        
        #st.subheader("Free Courses for the week")
        #free_courses = free_courses (10)
        #for row in free_courses.iterrows():

            #rec_title = row[1][0]
            #rec_url = row[1][1]
            #rec_price = row[1][2]
            #rec_num_sub = row[1][3]
            #stc.html(RESULT_TEMP2.format(rec_title,rec_url,rec_url,rec_num_sub),height=350)

    elif choice == "Recommend":
        
            

        cosine_sim_mat = vec_text_to_tfidf_mat(df['categories'])
        search_term = st.text_input("Search")
        x = st.sidebar.slider('Number of Recommendations',min_value = 5, max_value=30,step=1,value=15)
        num_of_rec = st.write("Total Recommendations:",x)
        num_of_rec=x
        if st.button("Recommend"):
            if search_term is not None:
                try:
                    results = get_recommendation_cat(search_term,cosine_sim_mat,df,num_of_rec)
                    #for row in results.iterrows():
                        #with st.expander("Results as JSON"):
                            #results_json = results.to_dict('index')
                            #st.write(results_json)
                    for row in results.iterrows():
                    	rec_imageurl = row[1][0]
                    	rec_title = row[1][1]
                    	rec_brand = row[1][2]
                    	rec_weight = row[1][3]
                    	rec_rating = row[1][4]
                    	rec_sim = row[1][5]
                    	stc.html(RESULT_TEMP3.format(rec_imageurl,rec_title,rec_brand,rec_weight,rec_rating,rec_sim),height=950)
                except:
                    results= "The search title didn't match any of the products, but there are some products you may like."
                    st.warning(results)
                    st.info("Suggested Options include")
                    result_df = search_term_if_not_found(search_term,df)
                    #st.dataframe(result_df)
                    for row in result_df.iterrows():
                    	rec_imageurl = row[1][0]
                    	rec_title = row[1][1]
                    	rec_brand = row[1][2]
                    	rec_weight = row[1][3]
                    	rec_rating = row[1][4]
                    	stc.html(RESULT_TEMP2.format(rec_imageurl,rec_title,rec_brand,rec_weight,rec_rating),height=950)
                    	

        
    elif choice == "User-based Recommend":

        cosine_sim_mat = vec_text_to_tfidf_mat(df['categories'])
        search_term2 = st.text_input("Enter User Id")
        name = prod_name(df,search_term2)
        x = st.sidebar.slider('Number of Recommendations',min_value = 5, max_value=30,step=1,value=15)
        num_of_rec = st.write("Total Recommendations:",x)
        num_of_rec=x
        if st.button("Recommend"):
            if name is not None:
                try:
                    results = get_recommendation_cat(name,cosine_sim_mat,df,num_of_rec)
                    #for row in results.iterrows():
                        #with st.expander("Results as JSON"):
                            #results_json = results.to_dict('index')
                            #st.write(results_json)
                    for row in results.iterrows():
                        rec_imageurl = row[1][0]
                        rec_title = row[1][1]
                        rec_brand = row[1][2]
                        rec_weight = row[1][3]
                        rec_rating = row[1][4]
                        rec_sim = row[1][5]
                        stc.html(RESULT_TEMP3.format(rec_imageurl,rec_title,rec_brand,rec_weight,rec_rating,rec_sim),height=950)
                except:
                    results= "This user id doesnot match any user."
                    st.warning(results)
                    
                    

        


    else:
        st.subheader("About")
        st.text("This is a short demo of an online course recommendation system. In this demo we have used cosine similarity score to find similar courses when the title is passed. And when there isn't any course in the database with the similar searched title, it recommends courses using the words present in the title.")


if __name__ == '__main__':
    main()