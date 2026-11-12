import streamlit as st
import cv2
import numpy as np
import math
import pandas as pd
from PIL import Image
import io

# Configuração da página
st.set_page_config(page_title="Spray Drop Analyser - Lote", page_icon="💧", layout="centered")

st.title("💧 Spray Drop Analyser (Análise em Lote)")
st.write("Faça o upload de várias fotos de papel hidrossensível para processar o lote com filtros personalizados e exportar em Excel.")

# --- BARRA LATERAL COM TODAS AS CONFIGURAÇÕES ---
st.sidebar.header("⚙️ Configurações de Análise")

tipo_papel = st.sidebar.selectbox(
    "Área Física de Referência", 
    ["Papel Padrão (19.75 cm²)", "Amostra Pequena (1.00 cm²)", "Personalizado (cm²)"]
)

if "Personalizado" in tipo_papel:
    area_fisica_cm2 = st.sidebar.number_input("Digite a área em cm²", min_value=0.1, max_value=100.0, value=19.75, step=0.25)
elif "1.00" in tipo_papel:
    area_fisica_cm2 = 1.00
else:
    area_fisica_cm2 = 19.75

canal_cor = st.sidebar.selectbox("Canal de Cor", ["Verde", "Azul", "Vermelho", "Cinza"], index=0)

sensibilidade_extra = st.sidebar.slider(
    "Ajuste Fino do Limiar (Offset)", 
    min_value=-50, 
    max_value=50, 
    value=0, 
    step=1
)

# Faixas de contagem para classificação
st.sidebar.subheader("📏 Faixas de Contagem (µm)")
limite_pequena = st.sidebar.number_input("Limite Máx. p/ Gotas Pequenas", min_value=50, max_value=300, value=150, step=10)
limite_media = st.sidebar.number_input("Limite Máx. p/ Gotas Médias", min_value=100, max_value=600, value=250, step=10)

# Filtros de Ruído (Crivo e Teto)
st.sidebar.subheader("🧹 Filtros de Ruído e Limites")
usar_crivo = st.sidebar.checkbox("Ativar Crivo Mínimo (remover poeira)", value=False)
diametro_crivo_min = 0.0
if usar_crivo:
    diametro_crivo_min = st.sidebar.number_input("Descartar gotas abaixo de (µm):", min_value=1.0, max_value=100.0, value=20.0, step=5.0)

usar_teto = st.sidebar.checkbox("Ativar Teto Máximo (remover borrões/uniões)", value=True)
diametro_teto_max = 800.0
if usar_teto:
    diametro_teto_max = st.sidebar.number_input("Descartar gotas acima de (µm):", min_value=300.0, max_value=2000.0, value=800.0, step=50.0)


# --- UPLOAD DE MÚLTIPLOS ARQUIVOS ---
st.markdown("---")
uploaded_files = st.file_uploader(
    "📂 Escolha as fotos do papel hidrossensível (várias de uma vez)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)
st.markdown("---")

if uploaded_files:
    st.info(f"📁 {len(uploaded_files)} imagem(ns) carregada(s) na fila de processamento.")
    
    if st.button("🚀 Processar Lote Completo e Gerar Relatório"):
        resultados_lote = []
        barra_progresso = st.progress(0)
        total_arquivos = len(uploaded_files)
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Abre a imagem
            image_pil = Image.open(uploaded_file).convert('RGB')
            img_original = np.array(image_pil)
            img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)

            altura_px, largura_px, _ = img_original.shape
            total_pixels_img = largura_px * altura_px

            # Seleção de Canal de Cor
            if canal_cor == 'Azul':
                img_canal = img_original[:, :, 0]
            elif canal_cor == 'Verde':
                img_canal = img_original[:, :, 1]
            elif canal_cor == 'Vermelho':
                img_canal = img_original[:, :, 2]
            else:
                img_canal = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

            # Otsu + Ajuste Fino
            limiar_otsu, _ = cv2.threshold(img_canal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            limiar_final = int(max(0, min(255, limiar_otsu + sensibilidade_extra)))
            _, img_binaria = cv2.threshold(img_canal, limiar_final, 255, cv2.THRESH_BINARY_INV)

            # Calibração Física
            area_total_um2 = area_fisica_cm2 * 100000000.0
            area_por_pixel_um2 = area_total_um2 / total_pixels_img if total_pixels_img > 0 else 1.0

            contornos, _ = cv2.findContours(img_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            total_gotas = 0
            pequenas = 0
            medias = 0
            grandes = 0
            soma_area_gotas_px = 0
            
            diametros_lista = []
            volumes_lista = []

            for c in contornos:
                area_px = cv2.contourArea(c)
                if area_px > 0:
                    area_um2 = area_px * area_por_pixel_um2
                    diametro_um = 2 * math.sqrt(area_um2 / math.pi)

                    # Aplica o crivo (filtro mínimo)
                    if usar_crivo and diametro_um < diametro_crivo_min:
                        continue
                    
                    # Aplica o teto (filtro máximo)
                    if usar_teto and diametro_um > diametro_
