import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image

# Configuração da página do App
st.set_page_config(page_title="Spray Drop Analyser", page_icon="💧", layout="centered")

st.title("💧 Spray Drop Analyser")
st.write("Faça o upload ou tire a foto do papel hidrossensível para realizar a análise automática.")

# 1. ÁREA DE UPLOAD / CÂMERA
st.markdown("---")
uploaded_file = st.file_uploader("📂 Escolha ou tire a foto do papel...", type=["jpg", "jpeg", "png"])
st.markdown("---")

# Barra lateral para os controlos avançados
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

# Ajuste fino do Limiar (-50 a +50)
sensibilidade_extra = st.sidebar.slider(
    "Ajuste Fino do Limiar (Offset)", 
    min_value=-50, 
    max_value=50, 
    value=0, 
    step=1,
    help="Esquerda (-) diminui o limiar (mais sensível). Direita (+) aumenta o limiar (menos sensível)."
)

# Configuração dos Limites de Tamanho para Contagem
st.sidebar.subheader("📏 Faixas de Contagem (µm)")
limite_pequena = st.sidebar.number_input("Limite Máx. p/ Gotas Pequenas", min_value=50, max_value=300, value=150, step=10)
limite_media = st.sidebar.number_input("Limite Máx. p/ Gotas Médias", min_value=100, max_value=600, value=250, step=10)

# Configuração do Crivo (Filtro de Ruído)
st.sidebar.subheader("🧹 Filtro de Crivo (Remoção de Ruído)")
usar_crivo = st.sidebar.checkbox("Ativar Crivo", value=False)
diametro_crivo_min = 0.0
if usar_crivo:
    diametro_crivo_min = st.sidebar.number_input("Descartar gotas com diâmetro abaixo de (µm):", min_value=1.0, max_value=100.0, value=20.0, step=5.0)

# Opção para exibir a imagem binarizada
mostrar_binaria = st.sidebar.checkbox("👁️ Exibir imagem binarizada", value=True)

# Processamento apenas se houver imagem carregada
if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert('RGB')
    
    st.subheader("🖼️ Imagem Carregada")
    st.image(image_pil, caption="Foto enviada", use_column_width=True)

    st.markdown("---")
    st.subheader("📊 Resultados da Análise")

    with st.spinner("A processar análise das gotas..."):
        # Converte a imagem para o formato do OpenCV
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

        # Otsu Automático + Ajuste Fino (-50 a +50)
        limiar_otsu, _ = cv2.threshold(img_canal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        limiar_final = int(limiar_otsu + sensibilidade_extra)
        limiar_final = max(0, min(255, limiar_final))
        
        _, img_binaria = cv2.threshold(img_canal, limiar_final, 255, cv2.THRESH_BINARY_INV)

        # Conversão de Área Física para Micrômetros (µm)
        area_total_um2 = area_fisica_cm2 * 100000000.0
        area_por_pixel_um2 = area_total_um2 / total_pixels_img if total_pixels_img > 0 else 1.0

        # Contornos
        contornos, _ = cv2.findContours(img_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_gotas = 0
        pequenas = 0    
        medias = 0      
        grandes = 0     
        soma_area_gotas_px = 0

        # Listas para cálculo volumétrico rigoroso
        diametros_lista = []
        volumes_lista = []

        for c in contornos:
            area_px = cv2.contourArea(c)
            if area_px > 0:
                area_um2 = area_px * area_por_pixel_um2
                diametro_um = 2 * math.sqrt(area_um2 / math.pi)

                # Aplica o crivo
                if usar_crivo and diametro_um < diametro_crivo_min:
                    continue

                soma_area_gotas_px += area_px
                total_gotas += 1

                # Cálculo exato do volume da esfera: V = (pi / 6) * D^3
                volume_gota = (math.pi / 6.0) * (diametro_um ** 3)
                
                diametros_lista.append(diametro_um)
                volumes_lista.append(volume_gota)

                # Classificação dinâmica
                if diametro_um < limite_pequena:
                    pequenas += 1
                elif limite_pequena <= diametro_um <= limite_media:
                    medias += 1
                else:
                    grandes += 1

        porcentagem_cobertura = (soma_area_gotas_px / total_pixels_img * 100.0) if total_pixels_img > 0 else 0.0

        # Cálculo dos Diâmetros Volumétricos (D0.1, DMV/D0.5, D0.9)
        d01, dmv, d09, span = 0.0, 0.0, 0.0, 0.0
        if total_gotas > 0:
            dados_ordenados = sorted(zip(diametros_lista, volumes_lista), key=lambda x: x[0])
            d_arr = np.array([x[0] for x in dados_ordenados])
            v_arr = np.array([x[1] for x in dados_ordenados])
            
            vol_acumulado = np.cumsum(v_arr)
            vol_total = vol_acumulado[-1]
            
            if vol_total > 0:
                vol_relativo = vol_acumulado / vol_total
                
                # Encontra os índices correspondentes aos percentis 10%, 50% e 90%
                idx_10 = np.searchsorted(vol_relativo, 0.10)
                idx_50 = np.searchsorted(vol_relativo, 0.50)
                idx_90 = np.searchsorted(vol_relativo, 0.90)
                
                d01 = d_arr[min(idx_10, len(d_arr)-1)]
                dmv = d_arr[min(idx_50, len(d_arr)-1)]
                d09 = d_arr[min(idx_90, len(d_arr)-1)]
                
                if dmv > 0:
                    span = (d09 - d01) / dmv

    if mostrar_binaria:
        st.subheader("🔍 Camada Binarizada")
        img_binaria_rgb = cv2.cvtColor(img_binaria, cv2.COLOR_GRAY2RGB)
        st.image(img_binaria_rgb, caption=f"Limiar Aplicado: {limiar_final} (Preto = Gotas, Branco = Fundo)", use_column_width=True)

    st.success("Análise concluída com sucesso!")
    
    # Métricas Gerais
    col1, col2 = st.columns(2)
    col1.metric("Total de Gotas", f"{total_gotas}")
    col2.metric("Área Coberta", f"{porcentagem_cobertura:.2f}%")

    # Métricas Volumétricas (DMV, D0.1, D0.9)
    st.subheader("🧪 Parâmetros Volumétricos (ASABE)")
    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    col_v1.metric("D0.1", f"{d01:.1f} µm")
    col_v2.metric("DMV (D0.5)", f"{dmv:.1f} µm")
    col_v3.metric("D0.9", f"{d09:.1f} µm")
    col_v4.metric("Amplitude (Span)", f"{span:.2f}")

    # Distribuição por Contagem
    st.subheader(f"📊 Contagem por Tamanho (Pequenas < {limite_pequena}µm | Médias até {limite_media}µm)")
    col3, col4, col5 = st.columns(3)
    col3.metric("Pequenas", f"{pequenas}")
    col4.metric("Médias", f"{medias}")
    col5.metric("Grandes", f"{grandes}")
    
    st.info(f"ℹ️ Área de Referência: {area_fisica_cm2} cm² | Limiar Otsu: {limiar_otsu:.1f} + Ajuste: {sensibilidade_extra} = {limiar_final}")
else:
    st.info("👈 Por favor, envie uma foto para iniciar a análise.")
