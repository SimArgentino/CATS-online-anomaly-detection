import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.image as img

from streamlit_option_menu import option_menu

# Function for the home page
def evaluation_page():
    with st.sidebar:
        st.title("Select parameter to compare")
        evaluated = option_menu(
            menu_title="Evaluate",
            options=["Architecture", 
                     "Optimizer",
                     "Percentile"],
            icons=[],
            menu_icon="pc-display",
            default_index=0,
        )


    # Comparison of Architectures
    match evaluated:
        case "Architecture":
            st.title("Model 1 vs Model 2")

            with st.sidebar:
                st.title("Select parameters to use")
                optimizer = option_menu(
                    menu_title="Optimizer",
                    options=["Adam",
                             "RMSprop"],
                    icons=[],
                    menu_icon="pc-display",
                    default_index=0,
                )
            
            with st.sidebar:
                epochs = option_menu(
                    menu_title="Epochs",
                    options=["30",
                             "60"],
                    icons=[],
                    menu_icon="pc-display",
                    default_index=0,
                )        
            
            with st.sidebar:
                threshold = option_menu(
                    menu_title="Percentile",
                    options=["90",
                             "95"],
                    icons=[],
                    menu_icon="pc-display",
                    default_index=0,
                )

            st.write("Selected parameters:")
            st.write("- Optimizer: " + str(optimizer))
            st.write("- Epochs: " + str(epochs))
            st.write("- Percentile: " + str(threshold))

            file_path1 = 'models_info/model1/' + str(optimizer) + '_e' + str(epochs) + '_p' + str(threshold) + '/'
            file_path2 = 'models_info/model2/' + str(optimizer) + '_e' + str(epochs) + '_p' + str(threshold) + '/'
            
            df1 = pd.read_csv(file_path1 + 'training_results.csv')
            df2 = pd.read_csv(file_path2 + 'training_results.csv')
    
            img1 = file_path1 + 'confusion_matrix.png'
            img2 = file_path2 + 'confusion_matrix.png'
    
            test1 = pd.read_csv(file_path1 + 'evaluation_metrics.csv')
            test2 = pd.read_csv(file_path2 + 'evaluation_metrics.csv')
    
            roc1 = file_path1 + 'roc_curve.png'
            roc2 = file_path2 + 'roc_curve.png'
    
            df1['Model'] = 'Model 1'
            df2['Model'] = 'Model 2'
            combined_df = pd.concat([df1, df2])
    
            # Select columns containing sensor values excluding "Epoch" and "Model"
            for value in df1.columns:
                if value not in ["Epoch", "Model"]:
                    fig = px.line(combined_df, x='Epoch', y=value, color='Model', title=str(value))
                    st.plotly_chart(fig)
            
            st.title("Confusion Matrix & Test Result")
    
            col1, col2 = st.columns(2)
    
            with col1:
                st.write("Model 1:")
                st.image(img1)
                st.write(test1)
                st.image(roc1)
    
            with col2:
                st.write("Model 2:")
                st.image(img2)
                st.write(test2)
                st.image(roc2)
        
        case "Optimizer":
            # Comparison of Optimizers
            st.title("Adam vs RMSprop")
    
            with st.sidebar:
                st.title("Select parameters to use")
                model = option_menu(
                    menu_title="Model",
                    options=["Model1",
                             "Model2"],
                    icons=[],
                    menu_icon="pc-display",
                    default_index=0,
                )
            
            with st.sidebar:
                epochs = option_menu(
                    menu_title="Epochs",
                    options=["30",
                             "60"],
                    icons=[],
                    menu_icon="pc-display",
                    default_index=0,
                )        
            
            with st.sidebar:
                threshold = option_menu(
                    menu_title="Percentile",
                    options=["90",
                             "95"],
                    icons=[],
                    menu_icon="pc-display",
                    default_index=0,
                )
            
            st.write("Selected parameters:")
            st.write("- Model: " + str(model))
            st.write("- Epochs: " + str(epochs))
            st.write("- Percentile: " + str(threshold))

            file_path1 = 'models_info/' + str(model) + '/Adam_e' + str(epochs) + '_p' + str(threshold) + '/'
            file_path2 = 'models_info/' + str(model) + '/RMSprop_e' + str(epochs) + '_p' + str(threshold) + '/'
            
            df1 = pd.read_csv(file_path1 + 'training_results.csv')
            df2 = pd.read_csv(file_path2 + 'training_results.csv')
    
            img1 = file_path1 + 'confusion_matrix.png'
            img2 = file_path2 + 'confusion_matrix.png'
    
            test1 = pd.read_csv(file_path1 + 'evaluation_metrics.csv')
            test2 = pd.read_csv(file_path2 + 'evaluation_metrics.csv')
    
            roc1 = file_path1 + 'roc_curve.png'
            roc2 = file_path2 + 'roc_curve.png'
    
            df1['Optimizer'] = 'Adam'
            df2['Optimizer'] = 'RMSprop'
            combined_df = pd.concat([df1, df2])
    
            # Select columns containing sensor values excluding "Epoch" and "Model"
            for value in df1.columns:
                if value not in ["Epoch", "Optimizer"]:
                    fig = px.line(combined_df, x='Epoch', y=value, color='Optimizer', title=str(value))
                    st.plotly_chart(fig)
            
            st.title("Confusion Matrix & Test Result")
            col1, col2 = st.columns(2)
    
            with col1:
                st.write("Adam:")
                st.image(img1)
                st.write(test1)
                st.image(roc1)
    
            with col2:
                st.write("RMSprop:")
                st.image(img2)
                st.write(test2)
                st.image(roc2)
    
        case "Percentile":
            # Comparison of Percentiles
            st.title("Threshold 90% vs Threshold 95%")
    
            with st.sidebar:
                st.title("Select parameters to use")
                model = option_menu(
                    menu_title="Model",
                    options=["Model1",
                             "Model2"],
                    icons=[],
                    menu_icon="pc-display",
                    default_index=0,
                )
    
            with st.sidebar:
                optimizer = option_menu(
                    menu_title="Optimizer",
                    options=["Adam",
                             "RMSprop"],
                    icons=[],
                    menu_icon="pc-display",
                    default_index=0,
                )
    
            with st.sidebar:
                epochs = option_menu(
                    menu_title="Epochs",
                    options=["30",
                             "60"],
                    icons=[],
                    menu_icon="pc-display",
                    default_index=0,
                )        
            
            st.write("Selected parameters:")
            st.write("- Model: " + str(model))
            st.write("- Optimizer: " + str(optimizer))
            st.write("- Epochs: " + str(epochs))

            file_path1 = 'models_info/' + str(model) + '/' + str(optimizer) + '_e' + str(epochs) + '_p90/'
            file_path2 = 'models_info/' + str(model) + '/' + str(optimizer) + '_e' + str(epochs) + '_p95/'
   
            df1 = pd.read_csv(file_path1 + 'training_results.csv')
            df2 = pd.read_csv(file_path2 + 'training_results.csv')
    
            img1 = file_path1 + 'confusion_matrix.png'
            img2 = file_path2 + 'confusion_matrix.png'
    
            test1 = pd.read_csv(file_path1 + 'evaluation_metrics.csv')
            test2 = pd.read_csv(file_path2 + 'evaluation_metrics.csv')        
    
            roc1 = file_path1 + 'roc_curve.png'
            roc2 = file_path2 + 'roc_curve.png'
    
            df1['Percentile'] = 'Threshold 90%'
            df2['Percentile'] = 'Threshold 95%'
            combined_df = pd.concat([df1, df2])
    
            # Select columns containing sensor values excluding "Epoch" and "Model"
            for value in df1.columns:
                if value not in ["Epoch", "Percentile"]:
                    fig = px.line(combined_df, x='Epoch', y=value, color='Percentile', title=str(value))
                    st.plotly_chart(fig)
            
            st.title("Confusion Matrix & Test Result")
    
            col1, col2 = st.columns(2)
    
            with col1:
                st.write("Threshold 90%:")
                st.image(img1)
                st.write(test1)
                st.image(roc1)
    
            with col2:
                st.write("Threshold 95%:")
                st.image(img2)
                st.write(test2)
                st.image(roc2)
