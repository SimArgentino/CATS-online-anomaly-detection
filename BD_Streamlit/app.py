import streamlit as st
import streamlit_option_menu
from streamlit_option_menu import option_menu

from menu_pages.home_page import home_page
from menu_pages.execution_page import execution_page
from menu_pages.model_page import model_page
from menu_pages.contact_page import contact_page
from menu_pages.colab_page import colab_page
from menu_pages.evaluation_page import evaluation_page

# Set page Layout
st.set_page_config(layout="wide")

# Sidebar Setup
with st.sidebar:
    selected = option_menu(
        menu_title = "Main Menu",
        options = ["Home", 
                   "About Project", 
                   "Code", 
                   "Evaluation", 
                   "Execution", 
                   "Contact"],
        icons = ["house", 
                 "book", 
                 "code-slash", 
                 "cpu", 
                 "graph-up-arrow", 
                 "envelope"],
        menu_icon = "cast",
        default_index = 0,
        )

# Pages switch
match selected:
    case "Home":
        home_page()
    case "About Project":
        model_page()
    case "Code":
        colab_page()
    case "Evaluation":
        evaluation_page()
    case "Execution":
        execution_page()
    case "Contact":
        contact_page()