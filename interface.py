import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Jumia Sentiment Analyzer", layout="wide")
st.title("Jumia Product Sentiment Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("jumia_dataset.csv")
    
    def map_sentiment(rating):
        rating_str = str(rating)
        if "1" in rating_str or "2" in rating_str:
            return "Bad"
        elif "3" in rating_str:
            return "Average"
        elif "4" in rating_str or "5" in rating_str:
            return "Good"
        else:
            return "Unknown"
            
    df['Sentiment'] = df['Review_Rating'].apply(map_sentiment)
    return df

@st.cache_resource
def load_models():
    try:
        model = joblib.load('logistic_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        return None, None

df = load_data()
model, vectorizer = load_models()

st.header("📊 Overall Dataset Sentiment")

total_reviews = len(df[df['Sentiment'] != 'Unknown'])
sentiment_counts = df['Sentiment'].value_counts()

good_count = sentiment_counts.get('Good', 0)
bad_count = sentiment_counts.get('Bad', 0)
avg_count = sentiment_counts.get('Average', 0)

good_pct = (good_count / total_reviews) * 100 if total_reviews > 0 else 0
bad_pct = (bad_count / total_reviews) * 100 if total_reviews > 0 else 0
avg_pct = (avg_count / total_reviews) * 100 if total_reviews > 0 else 0

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="🟢 Good Reviews", value=f"{good_count:,}", delta=f"{good_pct:.1f}% of total")
with col2:
    st.metric(label="🟡 Average Reviews", value=f"{avg_count:,}", delta=f"{avg_pct:.1f}% of total", delta_color="off")
with col3:
    st.metric(label="🔴 Bad Reviews", value=f"{bad_count:,}", delta=f"{bad_pct:.1f}% of total", delta_color="inverse")

st.divider()

st.header("🔍 Product Search Engine")

unique_products = df['Product_Name'].dropna().unique()
selected_product = st.selectbox("Search or select a product from the dataset:", unique_products)

if selected_product:
    product_df = df[df['Product_Name'] == selected_product]
    
    prod_info = product_df.iloc[0]
    
    st.subheader("Product Information")
    info_col1, info_col2 = st.columns([1, 3])
    
    with info_col1:
        if pd.notna(prod_info['Image_Link']) and prod_info['Image_Link'] != "No Image":
            st.image(prod_info['Image_Link'], use_container_width=True)
        else:
            st.write("No Image Available")
            
    with info_col2:
        st.write(f"**Category:** {prod_info['Category']}")
        st.write(f"**Current Price:** {prod_info['Current_Price']}")
        st.write(f"**Discount:** {prod_info['Discount']}")
        st.write(f"**Total Platform Ratings:** {prod_info['Total_Ratings']}")
        
    st.subheader(f"Scraped Reviews ({len(product_df)})")
    
    display_df = product_df[['Review_Rating', 'Sentiment', 'Review_Date', 'Verified_Purchase', 'Review_Text']]
    st.dataframe(display_df, use_container_width=True)

st.divider()

st.header("Sentiment Breakdown by Category")


df['Main_Category'] = df['Category'].apply(
    lambda x: x.split(" > ")[2] if isinstance(x, str) and len(x.split(" > ")) >= 3 else "Other"
)


cat_stats = pd.crosstab(df['Main_Category'], df['Sentiment'])

for s in ['Good', 'Average', 'Bad']:
    if s not in cat_stats:
        cat_stats[s] = 0

cat_stats['total'] = cat_stats['Good'] + cat_stats['Average'] + cat_stats['Bad']

cat_stats['good_percentage'] = ((cat_stats['Good'] / cat_stats['total']) * 100).round(2)
cat_stats['average_percentage'] = ((cat_stats['Average'] / cat_stats['total']) * 100).round(2)
cat_stats['bad_percentage'] = ((cat_stats['Bad'] / cat_stats['total']) * 100).round(2)

cat_stats = cat_stats.sort_values(by='total', ascending=False)

final_table = cat_stats[['good_percentage', 'average_percentage', 'bad_percentage', 'total']]

st.dataframe(final_table, use_container_width=True)

st.divider()

st.header("Live Model Prediction")
st.write("Test the Logistic Regression model on new text.")

user_input = st.text_area("Enter a custom product review:")

if st.button("Predict Sentiment"):
    if model and vectorizer:
        if user_input:
            vect_input = vectorizer.transform([user_input])
            prediction = model.predict(vect_input)[0]
            
            st.success(f"**The Logistic Regression Model predicts this review is:** {prediction.upper()}")
        else:
            st.warning("Please enter some text to predict.")
    else:
        st.error("Model files not found! Make sure 'logistic_model.pkl' and 'tfidf_vectorizer.pkl' are in the same folder as app.py.")