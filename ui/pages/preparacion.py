"""
Página de preparación de datos
"""
import streamlit as st
import pandas as pd
from src.data.data_loader import load_titanic_data
from src.data.preprocessor import TitanicPreprocessor
from src.visualization.plots import plot_train_test_split
from ui.styles.theme import render_header


def show():
    """Muestra la página de preparación de datos"""
    
    render_header(
        "🛠️ Preparación de Datos",
        "Limpieza, transformación y división de datos"
    )
    
    # Cargar datos
    with st.spinner("Cargando datos..."):
        df = load_titanic_data()
    
    if df is None:
        st.error("Error al cargar datos")
        return
    
    # Tabs para organizar contenido
    tabs = st.tabs([
        "🧹 Limpieza de Datos",
        "🔧 Feature Engineering",
        "📊 Codificación",
        "✂️ División Train/Test",
        "✅ Pipeline Completo"
    ])
    
    # Tab 1: Limpieza de Datos
    with tabs[0]:
        st.subheader("Limpieza de Datos")
        
        st.markdown("""
        Antes de entrenar modelos, necesitamos limpiar los datos:
        - Eliminar columnas innecesarias
        - Manejar valores nulos
        - Tratar outliers si es necesario
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Datos Originales:**")
            st.dataframe(df.head(5), use_container_width=True)
            st.caption(f"Shape: {df.shape}")
        
        with col2:
            st.markdown("**❌ Valores Nulos por Columna:**")
            null_df = pd.DataFrame({
                'Columna': df.columns,
                'Nulos': df.isnull().sum().values,
                'Porcentaje': (df.isnull().sum().values / len(df) * 100).round(2)
            })
            st.dataframe(null_df, use_container_width=True)
        
        st.markdown("---")
        
        # Estrategias de limpieza
        st.subheader("🔧 Estrategias de Limpieza")
        
        with st.expander("1️⃣ Eliminar Columnas Innecesarias"):
            st.markdown("""
            **Columnas a eliminar:**
            - `PassengerId`: ID único, no aporta información predictiva
            - `Name`: Nombres individuales, difícil de generalizar
            - `Ticket`: Número de ticket, no relevante para supervivencia
            - `Cabin`: Muchos valores nulos (77%), difícil de imputar
            """)
            
            st.code("""
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df_clean = df.drop(columns=columns_to_drop)
            """, language="python")
        
        with st.expander("2️⃣ Imputar Valores Nulos en Age"):
            st.markdown("""
            **Estrategia**: Usar la **mediana** de la edad
            - Menos sensible a outliers que la media
            - Mantiene la distribución central
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Edad Promedio", f"{df['Age'].mean():.1f} años")
            with col2:
                st.metric("Edad Mediana", f"{df['Age'].median():.1f} años")
            
            st.code("""
df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
            """, language="python")
        
        with st.expander("3️⃣ Imputar Valores Nulos en Embarked"):
            st.markdown("""
            **Estrategia**: Usar la **moda** (valor más frecuente)
            - Solo 2 valores nulos
            - La mayoría embarcó en Southampton (S)
            """)
            
            embarked_counts = df['Embarked'].value_counts()
            st.write("**Distribución de Embarked:**")
            st.write(embarked_counts)
            
            st.code("""
df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
            """, language="python")
        
        with st.expander("4️⃣ Imputar Valores Nulos en Fare"):
            st.markdown("""
            **Estrategia**: Usar la **mediana** de la tarifa
            - Solo 1 valor nulo
            - Evita distorsión por tarifas muy altas
            """)
            
            st.code("""
df_clean['Fare'].fillna(df_clean['Fare'].median(), inplace=True)
            """, language="python")
        
        # Aplicar limpieza
        if st.button("🧹 Aplicar Limpieza", type="primary"):
            preprocessor = TitanicPreprocessor()
            df_clean = preprocessor.clean_data(df)
            st.session_state['df_clean'] = df_clean
            
            st.success("✅ Datos limpiados exitosamente")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Filas Originales", len(df))
            with col2:
                st.metric("Filas Limpias", len(df_clean))
            
            st.dataframe(df_clean.head(), use_container_width=True)
    
    # Tab 2: Feature Engineering
    with tabs[1]:
        st.subheader("Feature Engineering")
        
        st.markdown("""
        **Feature Engineering** es el proceso de crear nuevas características o transformar
        las existentes para mejorar el rendimiento del modelo.
        """)
        
        st.info("""
        **💡 Features ya presentes útiles:**
        - `Pclass`: Clase del pasajero (1ra, 2da, 3ra)
        - `Sex`: Género del pasajero
        - `Age`: Edad del pasajero
        - `SibSp`: Número de hermanos/cónyuges a bordo
        - `Parch`: Número de padres/hijos a bordo
        - `Fare`: Tarifa pagada
        - `Embarked`: Puerto de embarque
        """)
        
        st.markdown("---")
        
        st.subheader("🎨 Ideas de Nuevas Features (Opcionales)")
        
        with st.expander("1️⃣ Family Size (Tamaño de Familia)"):
            st.markdown("""
            Combinar `SibSp` y `Parch` para obtener el tamaño total de la familia:
            
            ```python
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            ```
            
            **Hipótesis**: Familias de cierto tamaño podrían tener mayor probabilidad de supervivencia.
            """)
        
        with st.expander("2️⃣ IsAlone (Viaja Solo)"):
            st.markdown("""
            Indicador binario de si el pasajero viajaba solo:
            
            ```python
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
            ```
            
            **Hipótesis**: Pasajeros solos podrían tener diferentes tasas de supervivencia.
            """)
        
        with st.expander("3️⃣ Age Groups (Grupos de Edad)"):
            st.markdown("""
            Categorizar edades en grupos:
            
            ```python
            df['AgeGroup'] = pd.cut(df['Age'], 
                                     bins=[0, 12, 18, 35, 60, 100],
                                     labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
            ```
            
            **Hipótesis**: Diferentes grupos de edad tuvieron diferentes prioridades de evacuación.
            """)
        
        with st.expander("4️⃣ Fare Bins (Categorías de Tarifa)"):
            st.markdown("""
            Agrupar tarifas en categorías:
            
            ```python
            df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
            ```
            
            **Hipótesis**: El precio del ticket correlaciona con la clase y ubicación del camarote.
            """)
        
        st.warning("""
        **⚠️ Para este proyecto:**
        
        Usaremos las features originales para mantener la simplicidad y facilitar la interpretación.
        Las nuevas features pueden agregarse en iteraciones futuras del modelo.
        """)
    
    # Tab 3: Codificación
    with tabs[2]:
        st.subheader("Codificación de Variables Categóricas")
        
        st.markdown("""
        Los algoritmos de ML trabajan con números, por lo que debemos convertir
        variables categóricas a numéricas.
        """)
        
        if 'df_clean' not in st.session_state:
            st.warning("⚠️ Primero limpia los datos en la pestaña 'Limpieza de Datos'")
        else:
            df_clean = st.session_state['df_clean']
            
            st.markdown("### 🔤 Variables Categóricas a Codificar:")
            
            # Sex
            with st.expander("1️⃣ Sex (Género)"):
                st.markdown("**Label Encoding** - Convertir a binario:")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Antes:**")
                    st.write(df_clean['Sex'].value_counts())
                with col2:
                    st.write("**Después:**")
                    st.code("male → 0\nfemale → 1")
                
                st.code("""
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
                """, language="python")
            
            # Embarked
            with st.expander("2️⃣ Embarked (Puerto de Embarque)"):
                st.markdown("**One-Hot Encoding** - Crear columnas dummy:")
                
                st.write("**Antes:**")
                st.write(df_clean['Embarked'].value_counts())
                
                st.write("**Después:**")
                st.code("""
Embarked_C: [0, 1, 0, ...]
Embarked_Q: [0, 0, 1, ...]
Embarked_S: [1, 0, 0, ...]
                """)
                
                st.code("""
embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
df = pd.concat([df, embarked_dummies], axis=1)
df.drop('Embarked', axis=1, inplace=True)
                """, language="python")
            
            # Aplicar codificación
            if st.button("🔧 Aplicar Codificación", type="primary"):
                preprocessor = TitanicPreprocessor()
                df_encoded = preprocessor.encode_features(df_clean)
                st.session_state['df_encoded'] = df_encoded
                
                st.success("✅ Variables codificadas exitosamente")
                
                st.write("**Columnas después de codificación:**")
                st.write(df_encoded.columns.tolist())
                
                st.dataframe(df_encoded.head(), use_container_width=True)
    
    # Tab 4: División Train/Test
    with tabs[3]:
        st.subheader("División de Datos: Train/Test")
        
        st.markdown("""
        Dividimos los datos en dos conjuntos:
        - **Train (80%)**: Para entrenar el modelo
        - **Test (20%)**: Para evaluar el rendimiento en datos no vistos
        """)
        
        if 'df_encoded' not in st.session_state:
            st.warning("⚠️ Primero codifica las variables en la pestaña 'Codificación'")
        else:
            df_encoded = st.session_state['df_encoded']
            
            # Configuración de división
            col1, col2 = st.columns(2)
            
            with col1:
                test_size = st.slider(
                    "Porcentaje para Test",
                    min_value=10,
                    max_value=40,
                    value=20,
                    step=5
                ) / 100
            
            with col2:
                random_state = st.number_input(
                    "Random State",
                    min_value=0,
                    value=42
                )
            
            # Aplicar división
            if st.button("✂️ Dividir Datos", type="primary"):
                preprocessor = TitanicPreprocessor()
                X, y = preprocessor.prepare_features(df_encoded)
                X_train, X_test, y_train, y_test = preprocessor.split_data(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                # Guardar en session_state
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['feature_names'] = preprocessor.feature_names
                
                st.success("✅ Datos divididos exitosamente")
                
                # Mostrar estadísticas
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Train Set", len(X_train))
                with col2:
                    st.metric("Test Set", len(X_test))
                with col3:
                    st.metric("Features", len(preprocessor.feature_names))
                with col4:
                    st.metric("Train %", f"{(1-test_size)*100:.0f}%")
                
                # Visualización de la división
                st.plotly_chart(
                    plot_train_test_split(y_train, y_test),
                    use_container_width=True
                )
                
                st.info("""
                **✓ Balance de Clases:**
                
                Es importante que ambos conjuntos mantengan proporciones similares de
                supervivientes y no supervivientes. Usamos `stratify=y` para garantizar esto.
                """)
    
    # Tab 5: Pipeline Completo
    with tabs[4]:
        st.subheader("Pipeline Completo de Preprocesamiento")
        
        st.markdown("""
        Ejecuta todo el proceso de preparación de datos en un solo paso.
        """)
        
        if st.button("🚀 Ejecutar Pipeline Completo", type="primary", use_container_width=True):
            with st.spinner("Ejecutando pipeline..."):
                # Cargar datos
                df = load_titanic_data()
                
                # Crear preprocessor
                preprocessor = TitanicPreprocessor()
                
                # Pipeline completo
                X_train, X_test, y_train, y_test, df_clean = preprocessor.full_pipeline(df)
                
                # Guardar todo en session_state
                st.session_state['df_clean'] = df_clean
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['feature_names'] = preprocessor.feature_names
                st.session_state['data_prepared'] = True
                
                st.success("✅ Pipeline ejecutado exitosamente!")
                st.balloons()
                
                # Resumen
                st.markdown("### 📊 Resumen del Procesamiento")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Conjunto de Entrenamiento:**")
                    st.metric("Muestras", len(X_train))
                    st.write("Distribución de clases:")
                    st.write(pd.Series(y_train).value_counts())
                
                with col2:
                    st.markdown("**Conjunto de Prueba:**")
                    st.metric("Muestras", len(X_test))
                    st.write("Distribución de clases:")
                    st.write(pd.Series(y_test).value_counts())
                
                st.markdown("---")
                
                st.markdown("**Features utilizadas:**")
                st.write(preprocessor.feature_names)
                
                st.info("""
                ✓ **Datos listos para modelado!**
                
                Ahora puedes proceder a entrenar los modelos en las siguientes secciones:
                - 🌳 Árbol de Decisión (Harry)
                - 🌲 Random Forest (Tania)
                """)
    
    # Nota final
    st.markdown("---")
    st.markdown("""
    ## 📝 Resumen de Preparación
    
    **Pasos completados:**
    1. ✅ Limpieza de datos (eliminación de columnas, imputación de nulos)
    2. ✅ Codificación de variables categóricas
    3. ✅ División en conjuntos de entrenamiento y prueba
    4. ✅ Datos listos para modelado
    
    **Siguiente paso:** Entrena los modelos de Machine Learning
    """)