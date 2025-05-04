import re

import gspread
import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from gspread_dataframe import get_as_dataframe

# Page configuration
st.set_page_config(page_title="SphinxAI App", page_icon="📊", layout="wide")
st.title("SphinxAI")

# Create a connection to Google Sheets
@st.cache_resource
def connect_to_gsheets():
    # Create a connection object
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"]
    )
    client = gspread.authorize(credentials)
    return client

# Function to load data from Google Sheets
@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_data(sheet_url, sheet_name):
    client = connect_to_gsheets()
    sh = client.open_by_url(sheet_url)
    worksheet = sh.worksheet(sheet_name)
    df = get_as_dataframe(worksheet, evaluate_formulas=True, skiprows=0)
    # Clean up the DataFrame (remove empty rows and columns)
    df = df.dropna(how='all').dropna(axis=1, how='all')
    return df

# Function to check if a percentage value falls within a range string like "30-50%"
def is_in_percentage_range(value, range_str):
    if pd.isna(range_str):
        return False
    
    # Extract numbers from the range string
    numbers = re.findall(r'\d+', str(range_str))
    
    if len(numbers) == 1:
        # Single value like "50%"
        return value == int(numbers[0])
    elif len(numbers) == 2:
        # Range like "30-50%"
        lower = int(numbers[0])
        upper = int(numbers[1])
        return lower <= value <= upper
    else:
        return False

# Define the sheet names (outside of the sidebar for cleaner visibility)
sheet_url = "https://docs.google.com/spreadsheets/d/1zBrHhsj2ryzC3c6KwIw-9QZMOjiU4zYLlRFvQiGBukg/edit#gid=2040181610"
asymmetries_sheet = "BASE Asimetrias del Cerebro"
exercises_sheet = "BASE Ejercicios"

# # Set up the sidebar for configuration - hide URL input
# with st.sidebar:
#     st.header("Configuración")
#     # Hide the URL input but keep it functional
#     sheet_url = st.text_input(
#         "Google Sheet URL", 
#         value="https://docs.google.com/spreadsheets/d/1zBrHhsj2ryzC3c6KwIw-9QZMOjiU4zYLlRFvQiGBukg/edit#gid=2040181610",
#         label_visibility="collapsed"
#     )

# Load data when URL is provided
if sheet_url:
    try:
        with st.spinner("Cargando datos..."):
            # Load both sheets
            df_asymmetries = load_data(sheet_url, asymmetries_sheet)
            df_exercises = load_data(sheet_url, exercises_sheet)
            
            # Display original data in the main area
            st.header("Resultado del Test")
            
            # First, let's identify the "Escenarios" column
            escenarios_col = None
            for col in df_asymmetries.columns:
                if "Escenario" in col:
                    escenarios_col = col
                    break
            
            if not escenarios_col:
                escenarios_col = df_asymmetries.columns[-1]  # Default to last column
            
            # MOVE FILTERS TO SIDEBAR
            with st.sidebar:
                st.header("Filtros")
                
                # Create filters for each column except the Escenarios column
                filter_cols = [col for col in df_asymmetries.columns if col != escenarios_col and col != "Descripcion"]
                
                # Store filter selections
                filters = {}
                slider_filters = {}
                
                # Define which columns should use sliders instead of multiselect
                slider_columns = [
                    "% Hemisferio Correspondiente al ojo (Natural)",
                    "% de Hemisferio Recesivo (Anitnatural)"
                ]
                
                # Create filters
                for col_name in filter_cols:
                    if col_name in slider_columns:
                        # Create a single number slider filter
                        slider_filters[col_name] = st.slider(
                            f"Filtrar por {col_name}",
                            min_value=0,
                            max_value=100,
                            value=50,  # Default to middle value
                            step=1,
                            help=f"Selecciona un valor de porcentaje para {col_name}"
                        )
                    else:
                        # Create a regular multiselect filter
                        unique_values = sorted(df_asymmetries[col_name].dropna().unique())
                        filters[col_name] = st.multiselect(
                            f"Filtrar por {col_name}",
                            options=unique_values,
                            default=[]
                        )
            
            # Apply filters to get resulting Escenarios
            filtered_df = df_asymmetries.copy()
            any_filter_applied = False
            
            # Apply regular multiselect filters
            for col, selected_values in filters.items():
                if selected_values:  # If any values are selected for this column
                    any_filter_applied = True
                    filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
            
            # Apply slider filters
            for col, selected_value in slider_filters.items():
                # Default slider value is 50, so only apply filter if user has changed it
                if selected_value != 50:  # If slider has been adjusted from default
                    any_filter_applied = True
                    
                    # Create a boolean mask for rows where the selected value falls within the percentage range
                    mask = filtered_df.apply(
                        lambda row: (
                            # Check if the cell value is not NaN
                            not pd.isna(row[col]) and 
                            # Extract numbers from the range string
                            (
                                # Case 1: Single value like "50%"
                                (len(re.findall(r'\d+', str(row[col]))) == 1 and 
                                 selected_value == int(re.findall(r'\d+', str(row[col]))[0])) or
                                # Case 2: Range like "30-50%"
                                (len(re.findall(r'\d+', str(row[col]))) == 2 and 
                                 int(re.findall(r'\d+', str(row[col]))[0]) <= selected_value <= int(re.findall(r'\d+', str(row[col]))[1]))
                            )
                        ),
                        axis=1
                    )
                    
                    filtered_df = filtered_df[mask]
            
            # Get resulting Escenarios without displaying them
            resulting_escenarios = filtered_df[escenarios_col].dropna().unique() if any_filter_applied else df_asymmetries[escenarios_col].dropna().unique()
            
            # Display the Escenarios that will be used for filtering
            if 'resulting_escenarios' in locals():
                # Move scenario selection to sidebar
                with st.sidebar:
                    st.header("Selección de Escenarios")
                    # Allow selecting from all available escenarios if no filters are applied
                    all_escenarios = df_asymmetries[escenarios_col].dropna().unique()
                    selected_escenarios = st.multiselect(
                        "Selecciona los Escenarios:",
                        options=all_escenarios,  # Use all available escenarios
                        default=resulting_escenarios if len(resulting_escenarios) <= 5 else resulting_escenarios[:5]
                    )
                
                # If no escenarios are selected and no filters applied, don't show any exercises
                if not selected_escenarios:
                    st.warning("Por favor, selecciona al menos un Escenario para filtrar los ejercicios.")
                    st.stop()
                
                # Filter the asymmetries dataframe based on selected scenarios
                filtered_df = df_asymmetries[df_asymmetries[escenarios_col].isin(selected_escenarios)]
                
                # Show filtered asymmetries data
                st.subheader("Datos Filtrados")
                st.dataframe(filtered_df)
                
                if selected_escenarios:
                    # Find all columns that start with "Escenario:"
                    escenario_columns = [col for col in df_exercises.columns if str(col).startswith("Escenario:")]
                    
                    # Create a mask to filter rows
                    mask = pd.Series(False, index=df_exercises.index)
                    
                    # For each Escenario column, check if any selected value appears in it
                    for col in escenario_columns:
                        # Convert column to string to ensure comparison works
                        df_exercises[col] = df_exercises[col].astype(str)
                        
                        # For each selected scenario code
                        for escenario_code in selected_escenarios:
                            # Find rows where this code appears in this column
                            column_mask = df_exercises[col].str.contains(escenario_code, na=False)
                            # Update the overall mask
                            mask = mask | column_mask
                    
                    # Apply the mask to filter the exercises dataframe
                    filtered_exercises = df_exercises[mask]
                    
                    # Display the filtered data
                    if not filtered_exercises.empty:
                        st.header("Ejercicios")
                        
                        # Add filters for exercises table
                        col1, col2 = st.columns(2)
                        with col1:
                            # Filter by Filmina
                            filminas = sorted(filtered_exercises['Filmina'].unique())
                            selected_filminas = st.multiselect(
                                'Filtrar por Filmina',
                                options=filminas,
                                default=[]
                            )
                            
                            # Filter by Nivel de Ejercicio
                            niveles = sorted(filtered_exercises['Nivel de Ejercicio'].unique())
                            selected_niveles = st.multiselect(
                                'Filtrar por Nivel de Ejercicio',
                                options=niveles,
                                default=[]
                            )
                        
                        with col2:
                            # Filter by Dificultad
                            dificultades = sorted(filtered_exercises['Dificultad'].unique())
                            selected_dificultades = st.multiselect(
                                'Filtrar por Dificultad',
                                options=dificultades,
                                default=[]
                            )
                            
                            # Filter by Dificultad de Nivel Ejercicio
                            dif_niveles = sorted(filtered_exercises['Dificultad de Nivel Ejercicio'].unique())
                            selected_dif_niveles = st.multiselect(
                                'Filtrar por Dificultad de Nivel Ejercicio',
                                options=dif_niveles,
                                default=[]
                            )
                        
                        # Apply filters to exercises table
                        if selected_filminas:
                            filtered_exercises = filtered_exercises[filtered_exercises['Filmina'].isin(selected_filminas)]
                        if selected_dificultades:
                            filtered_exercises = filtered_exercises[filtered_exercises['Dificultad'].isin(selected_dificultades)]
                        if selected_niveles:
                            filtered_exercises = filtered_exercises[filtered_exercises['Nivel de Ejercicio'].isin(selected_niveles)]
                        if selected_dif_niveles:
                            filtered_exercises = filtered_exercises[filtered_exercises['Dificultad de Nivel Ejercicio'].isin(selected_dif_niveles)]
                        
                        # Display filtered exercises
                        st.dataframe(filtered_exercises)
                        
                        # Export option
                        st.download_button(
                            label="Descargar datos filtrados como CSV",
                            data=filtered_exercises.to_csv(index=False).encode('utf-8'),
                            file_name='filtered_exercises.csv',
                            mime='text/csv',
                        )
                    else:
                        st.warning("No se encontraron ejercicios que coincidan con los criterios seleccionados.")
            else:
                st.warning("Por favor, aplica filtros en la tabla de Resultados del Test primero para generar Escenarios.")
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error("Make sure the Google Sheet is accessible and the correct service account is set up.")
