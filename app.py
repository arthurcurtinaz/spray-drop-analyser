import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image

# Configuração da página do App
st.set_page_config(page_title="App de Análise de Gotas", page_icon="💧", layout="centered")

st.title("💧 Análise de Gotas - Protótipo Mobile")
st.write("Faça o upload da foto do papel hidrossensível para começar a análise.")

# 1. ÁREA DE UPLOAD
st.markdown("---")
uploaded_file = st.file_uploader("📂 Escolha a foto do papel hidrossensível...", type=["jpg", "jpeg", "png"])
st.markdown("---")

# Barra lateral para os controles avançados
st.sidebar.header("⚙️ Configurações de Análise")

# Seleção da Área Física de Referência
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
sensibilidade_extra = st.sidebar.slider("Sensibilidade Extra (Otsu)", min_value=0, max_value=40, value=15, step=1)

# Configuração dos Limites de Tamanho
st.sidebar.subheader("📏 Faixas de Tamanho (µm)")
limite_pequena = st.sidebar.number_input("Limite Máx. p/ Gotas Pequenas", min_value=50, max_value=300, value=150, step=10)
limite_media = st.sidebar.number_input("Limite Máx. p/ Gotas Médias", min_value=100, max_value=600, value=250, step=10)

# Configuração do Crivo (Agora baseado no Diâmetro em µm para facilitar o ajuste)
st.sidebar.subheader("🧹 Filtro de Crivo (Remoção de Ruído)")
usar_crivo = st.sidebar.checkbox("Ativar Crivo", value=False)
diametro_crivo_min = 0.0
if usar_crivo:
    diametro_crivo_min = st.sidebar.number_input("Descartar gotas com diâmetro abaixo de (µm):", min_value=1.0, max_value=100.0, value=20.0, step=5.0)

# Opção para exibir a imagem binarizada
mostrar_binaria = st.sidebar.checkbox("👁️ Exibir imagem binarizada (estilo QGIS)", value=True)

# Processamento apenas se houver imagem carregada
if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    img_original = np.array(image_pil)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)

    st.image(image_pil, caption="Imagem Original Carregada", use_column_width=True)
    
    with st.spinner("Processando análise das gotas..."):
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

        # Otsu Automático + Sensibilidade Extra
        limiar_otsu, _ = cv2.threshold(img_canal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        limiar_final = int(limiar_otsu + sensibilidade_extra)
        _, img_binaria = cv2.threshold(img_canal, limiar_final, 255, cv2.THRESH_BINARY_INV)

        # Conversão de Área Física para Micrômetros (µm)
        area_total_um2 = area_fisica_cm2 * 100000000.0
        area_por_pixel_um2 = area_total_um2 / total_pixels_img

        # Contornos
        contornos, _ = cv2.findContours(img_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_gotas = 0
        pequenas = 0    
        medias = 0      
        grandes = 0     
        soma_area_gotas_px = 0

        for c in contornos:
            area_px = cv2.contourArea(c)
            if area_px > 0:
                area_um2 = area_px * area_por_pixel_um2
                diametro_um = 2 * math.sqrt(area_um2 / math.pi)

                # Aplica o crivo: se ativado e o diâmetro for menor que o corte, ignora a gota
                if usar_crivo and diametro_um < diametro_crivo_min:
                    continue

                soma_area_gotas_px += area_px
                total_gotas += 1

                # Classificação dinâmica baseada nos inputs do usuário
                if diametro_um < limite_pequena:
                    pequenas += 1
                elif limite_pequena <= diametro_um <= limite_media:
                    medias += 1
                else:
                    grandes += 1

        porcentagem_cobertura = (soma_area_gotas_px / total_pixels_img) * 100.0

    if mostrar_binaria:
        st.subheader("🔍 Camada Binarizada (Preto e Branco)")
        img_binaria_rgb = cv2.cvtColor(img_binaria, cv2.COLOR_GRAY2RGB)
        st.image(img_binaria_rgb, caption=f"Limiar Aplicado: {limiar_final} (Preto = Gotas, Branco = Fundo)", use_column_width=True)

    st.success("Análise concluída com sucesso!")
    
    col1, col2 = st.columns(2)
    col1.metric("Total de Gotas", f"{total_gotas}")
    col2.metric("Área Coberta", f"{porcentagem_cobertura:.2f}%")

    st.subheader(f"📊 Distribuição por Tamanho (Pequenas < {limite_pequena}µm | Médias até {limite_media}µm)")
    col3, col4, col5 = st.columns(3)
    col3.metric("Pequenas", f"{pequenas}")
    col4.metric("Médias", f"{medias}")
    col5.metric("Grandes", f"{grandes}")
    
    st.info(f"ℹ️ Área: {area_fisica_cm2} cm² | Limiar Otsu: {limiar_otsu:.1f} + Sensibilidade: {sensibilidade_extra} = {limiar_final}")
else:
    st.info("👈 Por favor, envie uma imagem na opção acima para iniciar a análise.")