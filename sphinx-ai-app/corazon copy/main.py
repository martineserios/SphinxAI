import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# Page config and imports
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Page config
st.set_page_config(
    page_title="Panel de Ejercicios",
    page_icon="🏃‍♂️",
    layout="wide"
)

# Debug mode toggle
show_debug = st.sidebar.toggle('Mostrar información de debug', value=False)

# Function to log debug info
def debug_log(message):
    if show_debug:
        st.write(f"Debug: {message}")

# Function to initialize Google Sheets API
def init_google_sheets():
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
        )
        service = build('sheets', 'v4', credentials=credentials)
        return service
    except Exception as e:
        st.error(f"Error initializing Google Sheets API: {str(e)}")
        return None

# Function to load data from Google Sheets
@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_gsheets_data():
    try:
        debug_log("Starting to read Google Sheets")
        service = init_google_sheets()
        if not service:
            return None
            
        sheet = service.spreadsheets()
        SHEET_ID = st.secrets["sheet_id"]
        RANGE_NAME = st.secrets["sheet_range"]
        
        result = sheet.values().get(
            spreadsheetId=SHEET_ID,
            range=RANGE_NAME
        ).execute()
        
        values = result.get('values', [])
        
        if not values:
            st.error('No data found in the sheet.')
            return None
            
        df = pd.DataFrame(values[1:], columns=values[0])
        debug_log("Successfully read Google Sheets data")
        return process_dataframe(df)
        
    except Exception as e:
        st.error(f"Error accessing Google Sheets: {str(e)}")
        debug_log(f"Full error details: {str(e)}")
        return None

# Function to process dataframe (common operations)
def process_dataframe(df):
    try:
        debug_log(f"Columns in dataframe: {df.columns.tolist()}")
        
        # Convert Nivel del Ejercicio to numeric (removing % if present)
        if 'Nivel del Ejercicio' in df.columns:
            debug_log("Processing Nivel del Ejercicio column")
            try:
                df['Nivel del Ejercicio'] = df['Nivel del Ejercicio'].str.rstrip('%').astype('float') / 100.0
                debug_log("Successfully converted Nivel del Ejercicio")
            except Exception as e:
                debug_log(f"Error converting Nivel del Ejercicio: {str(e)}")
                debug_log(f"First few values of Nivel del Ejercicio: {df['Nivel del Ejercicio'].head()}")
        
        return df
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        debug_log(f"Full error details: {str(e)}")
        debug_log("DataFrame info:")
        if show_debug:
            st.write(df.info())
        return None

# Function to load data from CSV
@st.cache_data
def load_csv_data(uploaded_file):
    try:
        debug_log("Starting to read CSV file")
        df = pd.read_csv(uploaded_file)
        debug_log("Successfully read CSV file")
        debug_log(f"CSV columns: {df.columns.tolist()}")
        return process_dataframe(df)
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        debug_log(f"Full error details: {str(e)}")
        return None

# Data source selection
st.sidebar.title("Fuente de Datos")
data_source = st.sidebar.radio(
    "Seleccionar fuente de datos:",
    ["Cargar CSV", "Google Sheets"]
)

# Load data based on selected source
if data_source == "Cargar CSV":
    uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV", type="csv")
    if uploaded_file is not None:
        df = load_csv_data(uploaded_file)
    else:
        st.info("Por favor, carga un archivo CSV")
        st.stop()
else:  # Google Sheets
    if 'gcp_service_account' not in st.secrets:
        st.error("""
        No se encontró la configuración de Google Sheets. 
        
        Para usar Google Sheets, necesitas:
        1. Crear un archivo `.streamlit/secrets.toml` con:
           ```toml
           [gcp_service_account]
           type = "service_account"
           project_id = "tu-project-id"
           private_key_id = "tu-private-key-id"
           private_key = "tu-private-key"
           client_email = "tu-client-email"
           client_id = "tu-client-id"
           auth_uri = "https://accounts.google.com/o/oauth2/auth"
           token_uri = "https://oauth2.googleapis.com/token"
           auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
           client_x509_cert_url = "tu-cert-url"
           
           # Detalles del Google Sheet
           sheet_id = "tu-sheet-id"
           sheet_range = "Sheet1!A1:J1000"  # Ajusta el rango según necesites
           ```
        2. Compartir el Google Sheet con el email del service account
        """)
        st.stop()
    else:
        df = load_gsheets_data()

if df is not None:
    # Debug information
    if show_debug:
        with st.sidebar.expander("Debug Information"):
            st.write({
                "Shape": df.shape,
                "Columns": df.columns.tolist(),
                "Data Types": df.dtypes.to_dict()
            })
    
    # Sidebar filters
    st.sidebar.title("Filtros")

    # Create filters for all columns
    selected_filters = {}
    
    # Create two columns in sidebar for filters to save space
    filter_cols = st.sidebar.columns(2)
    
    # Add clear all filters button
    if st.sidebar.button("Limpiar todos los filtros"):
        st.experimental_rerun()
    
    # Distribute filters between the two columns
    for i, column in enumerate(df.columns):
        # Skip certain columns that might not need filtering
        if column not in ['Nivel del Ejercicio']:  # Add any columns you want to exclude
            with filter_cols[i % 2]:  # Alternate between columns
                # Get unique values and handle NaN
                unique_values = df[column].unique()
                # Replace NaN with a string representation if needed
                unique_values = ['Sin valor' if pd.isna(x) else x for x in unique_values]
                # Sort values, putting 'Sin valor' at the end if it exists
                unique_values = sorted([x for x in unique_values if x != 'Sin valor']) + \
                              (['Sin valor'] if 'Sin valor' in unique_values else [])
                
                selected_filters[column] = st.multiselect(
                    f"Filtrar {column}",
                    options=unique_values,
                    default=None,  # Start with no selection
                    key=f"filter_{column}"  # Unique key for each filter
                )

    # Apply filters
    mask = pd.Series(True, index=df.index)
    active_filters = {}

    for col, selected in selected_filters.items():
        if selected:  # Only apply filter if something is selected
            filter_values = selected.copy()
            if 'Sin valor' in selected:
                filter_values.remove('Sin valor')
                col_mask = ((df[col].isin(filter_values)) | (df[col].isna()))
            else:
                col_mask = (df[col].isin(filter_values))
            mask = mask & col_mask
            active_filters[col] = selected

    filtered_df = df[mask]

    # Show active filters summary
    if active_filters:
        st.sidebar.markdown("### Filtros Activos")
        for col, values in active_filters.items():
            st.sidebar.write(f"**{col}:** {', '.join(values)}")
    else:
        st.sidebar.markdown("### No hay filtros activos")

    # Main content
    st.title("Panel de Ejercicios 🏃‍♂️")

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Ejercicios", len(filtered_df))
    with col2:
        st.metric("Tipos de Ejercicios", filtered_df["Ejercicios"].nunique())
    with col3:
        st.metric("Niveles de Dificultad", filtered_df["Dificultad"].nunique())

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Visualizaciones", "📋 Datos", "📝 Análisis"])

    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution by Prioridad Coordinativa
            fig1 = px.pie(
                filtered_df,
                names="Prioridad Coordinativa",
                title="Distribución por Prioridad Coordinativa"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Distribution by Hemisferio
            hemisferio_counts = filtered_df["Hemisferio"].value_counts().reset_index()
            hemisferio_counts.columns = ['Hemisferio', 'Cantidad']
            fig2 = px.bar(
                hemisferio_counts,
                x="Hemisferio",
                y="Cantidad",
                title="Distribución por Hemisferio"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Additional visualization for difficulty levels
        fig3 = px.bar(
            filtered_df.groupby(["Dificultad", "Hemisferio"]).size().reset_index(name="count"),
            x="Dificultad",
            y="count",
            color="Hemisferio",
            title="Ejercicios por Dificultad y Hemisferio",
            barmode="group"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        # Show filtered dataframe with column selection
        columns_to_display = st.multiselect(
            "Seleccionar columnas para mostrar:",
            options=df.columns,
            default=list(df.columns)
        )
        
        st.dataframe(
            filtered_df[columns_to_display],
            use_container_width=True,
            hide_index=True
        )

    with tab3:
        # Analysis section
        st.subheader("Análisis de Ejercicios")
        
        # Summary by difficulty
        st.write("### Resumen por Nivel de Dificultad")
        diff_summary = filtered_df.groupby("Dificultad").agg({
            "Ejercicios": "count",
            "Nivel del Ejercicio": "mean"
        }).round(2)
        st.dataframe(diff_summary)
        
        # Summary by hemisphere and coordination priority
        st.write("### Distribución por Hemisferio y Prioridad Coordinativa")
        hemi_coord = pd.crosstab(
            filtered_df["Hemisferio"],
            filtered_df["Prioridad Coordinativa"]
        )
        st.dataframe(hemi_coord)

    # Footer with download button and filter summary
    st.write("### Resumen de Filtros Aplicados")
    filter_summary = {}
    for col, selected in selected_filters.items():
        if set(selected) != set(df[col].unique()):  # Only show if filter is active
            filter_summary[col] = selected
    
    if filter_summary:
        st.json(filter_summary)
    else:
        st.write("No hay filtros activos - mostrando todos los datos")
    
    st.download_button(
        label="Descargar datos filtrados",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name='ejercicios_filtrados.csv',
        mime='text/csv',
    )