import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Titanic EDA Pro", layout="wide")
sns.set_theme(style="whitegrid")

# --- 2. DATA UTILITIES ---
@st.cache_data # Caches data so it doesn't reload on every click
def load_and_clean_data(file):
    """Loads CSV and fixes type issues (like the PyArrow error)."""
    # Load and treat common empty values as NaN
    df = pd.read_csv(file, na_values=['', ' ', 'NA', 'N/A'])
    
    # Fix PassengerId/Numeric issues
    if 'PassengerId' in df.columns:
        df['PassengerId'] = pd.to_numeric(df['PassengerId'], errors='coerce')
        df = df.dropna(subset=['PassengerId'])
        df['PassengerId'] = df['PassengerId'].astype(int)
    
    # Simple cleanup: Fill Age with median for visualizations
    df['Age'] = df['Age'].fillna(df['Age'].median())
    return df

# --- 3. COMPONENT FUNCTIONS (Visualizations) ---
def render_overview(df):
    st.header("📊 Dataset Overview")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Sample Data")
        st.dataframe(df.head(10))
    with col2:
        st.subheader("Missing Value Check")
        st.write(df.isnull().sum())

def render_proportions(df):
    st.header("🍕 Categorical Proportions")
    cols = st.columns(3)
    
    # Helper to plot pie charts
    def plot_pie(column, labels, colors, title, ax_obj):
        counts = df[column].value_counts()
        ax_obj.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
        ax_obj.set_title(title)

    fig1, ax1 = plt.subplots()
    plot_pie('Survived', ['Dead', 'Survived'], ['#ff9999','#66b3ff'], "Survival", ax1)
    cols[0].pyplot(fig1)

    fig2, ax2 = plt.subplots()
    plot_pie('Sex', df['Sex'].unique(), ['#c2c2f0','#ffb3e6'], "Gender", ax2)
    cols[1].pyplot(fig2)

    fig3, ax3 = plt.subplots()
    plot_pie('Pclass', [f'Class {i}' for i in df['Pclass'].unique()], None, "Class", ax3)
    cols[2].pyplot(fig3)

def render_survival_analysis(df):
    st.header("🧬 Survival Determinants")
    category = st.selectbox("View survival rate by:", ["Sex", "Pclass", "Embarked", "SibSp"])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=category, y='Survived', data=df, palette='magma', ax=ax)
    st.pyplot(fig)

# --- 4. MAIN APP LOGIC ---
def main():
    st.title("🚢 Titanic Exploratory Data Analysis")
    
    # Sidebar
    st.sidebar.title("Settings")
    uploaded_file = st.sidebar.file_uploader("Upload Titanic CSV", type="csv")
    
    if uploaded_file:
        df = load_and_clean_data(uploaded_file)
    else:
        # Fallback to local file
        try:
            df = load_and_clean_data('titanic_data.csv')
            st.sidebar.info("Using default: titanic_data.csv")
        except:
            st.warning("Waiting for data upload...")
            return

    # Navigation
    menu = ["Data Overview", "Proportions", "Survival Factors", "Correlation"]
    choice = st.sidebar.selectbox("Navigate To:", menu)

    if choice == "Data Overview":
        render_overview(df)
    elif choice == "Proportions":
        render_proportions(df)
    elif choice == "Survival Factors":
        render_survival_analysis(df)
    elif choice == "Correlation":
        st.header("🔗 Relationship Heatmap")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

if __name__ == "__main__":
    main()