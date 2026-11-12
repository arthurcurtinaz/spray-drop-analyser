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
st.write("Faça o upload de várias fotos de papel hidrossensível para processar o lote e exportar em Excel.")

# Barra lateral para configurações globais do lote
st.sidebar.header("⚙️ Configurações do Lote")
area_fisica_cm2 = st.sidebar.number_input("Área Física de Referência (cm²)", value=19.75)
canal_cor = st.sidebar.selectbox("Canal de Cor", ["Verde", "Azul", "Vermelho", "Cinza"], index=0)
sensibilidade_extra = st.sidebar.slider("Ajuste Fino do Limiar (Offset)", -50, 50, 0)
diametro_teto_max = st.sidebar.number_input("Teto Máximo (µm)", value=800.0)

# UPLOAD DE MÚLTIPLOS ARQUIVOS
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
        
        # Barra de progresso visual no Streamlit
        barra_progresso = st.progress(0)
        total_arquivos = len(uploaded_files)
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Abre a imagem
            image_pil = Image.open(uploaded_file).convert('RGB')
            img_original = np.array(image_pil)
            img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)

            altura_px, largura_px, _ = img_original.shape
            total_pixels_img = largura_px * altura_px

            # Canal de cor
            if canal_cor == 'Azul':
                img_canal = img_original[:, :, 0]
            elif canal_cor == 'Verde':
                img_canal = img_original[:, :, 1]
            elif canal_cor == 'Vermelho':
                img_canal = img_original[:, :, 2]
            else:
                img_canal = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

            # Otsu + Offset
            limiar_otsu, _ = cv2.threshold(img_canal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            limiar_final = int(max(0, min(255, limiar_otsu + sensibilidade_extra)))
            _, img_binaria = cv2.threshold(img_canal, limiar_final, 255, cv2.THRESH_BINARY_INV)

            # Calibração e Contornos
            area_total_um2 = area_fisica_cm2 * 100000000.0
            area_por_pixel_um2 = area_total_um2 / total_pixels_img if total_pixels_img > 0 else 1.0

            contornos, _ = cv2.findContours(img_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            total_gotas = 0
            soma_area_gotas_px = 0
            diametros_lista = []
            volumes_lista = []

            for c in contornos:
                area_px = cv2.contourArea(c)
                if area_px > 0:
                    area_um2 = area_px * area_por_pixel_um2
                    diametro_um = 2 * math.sqrt(area_um2 / math.pi)

                    if diametro_um <= diametro_teto_max: # Filtro de teto
                        soma_area_gotas_px += area_px
                        total_gotas += 1
                        volume_gota = (math.pi / 6.0) * (diametro_um ** 3)
                        diametros_lista.append(diametro_um)
                        volumes_lista.append(volume_gota)

            porcentagem_cobertura = (soma_area_gotas_px / total_pixels_img * 100.0) if total_pixels_img > 0 else 0.0

            # Cálculos Volumétricos
            d01, dmv, d09, span = 0.0, 0.0, 0.0, 0.0
            if total_gotas > 0:
                dados_ordenados = sorted(zip(diametros_lista, volumes_lista), key=lambda x: x[0])
                d_arr = np.array([x[0] for x in dados_ordenados])
                v_arr = np.array([x[1] for x in dados_ordenados])
                
                vol_acumulado = np.cumsum(v_arr)
                vol_total = vol_acumulado[-1]
                
                if vol_total > 0:
                    vol_relativo = vol_acumulado / vol_total
                    idx_10 = np.searchsorted(vol_relativo, 0.10)
                    idx_50 = np.searchsorted(vol_relativo, 0.50)
                    idx_90 = np.searchsorted(vol_relativo, 0.90)
                    
                    d01 = d_arr[min(idx_10, len(d_arr)-1)]
                    dmv = d_arr[min(idx_50, len(d_arr)-1)]
                    d09 = d_arr[min(idx_90, len(d_arr)-1)]
                    
                    if dmv > 0:
                        span = (d09 - d01) / dmv

            # Salva os resultados desta imagem na lista do lote
            resultados_lote.append({
                "Arquivo": uploaded_file.name,
                "Total de Gotas": total_gotas,
                "Cobertura (%)": round(porcentagem_cobertura, 2),
                "D0.1 (µm)": round(d01, 1),
                "DMV / D0.5 (µm)": round(dmv, 1),
                "D0.9 (µm)": round(d09, 1),
                "Span": round(span, 2)
            })

            # Atualiza barra de progresso
            barra_progresso.progress((i + 1) / total_arquivos)

        # Converte a lista em DataFrame do Pandas
        df_resultado = pd.DataFrame(resultados_lote)
        
        st.success("✅ Processamento do lote concluído com sucesso!")
        st.dataframe(df_resultado)

        # CRIAR ARQUIVO EXCEL PARA DOWNLOAD
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_resultado.to_excel(writer, index=False, sheet_name='Analise_Gotas')
        processed_data = output.getvalue()

        st.download_button(
            label="📥 Baixar Relatório Consolidado em Excel (.xlsx)",
            data=processed_data,
            file_name="relatorio_gotas_lote.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("👈 Selecione várias fotos na caixa acima para iniciar o processamento em lote.")
