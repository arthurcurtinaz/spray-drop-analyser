import streamlit as st
import cv2
import numpy as np
import math
import pandas as pd
from PIL import Image
import io

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Spray Drop Analyser", 
    page_icon="💧", 
    layout="wide"
)

# --- CONFIGURAÇÃO DE PWA PARA INSTALAÇÃO DIRETA ---
pwa_code = """
<link rel="manifest" href="data:application/manifest+json,{
  &quot;name&quot;: &quot;Spray Drop Analyser&quot;,
  &quot;short_name&quot;: &quot;SprayDrop&quot;,
  &quot;start_url&quot;: &quot;.&quot;,
  &quot;display&quot;: &quot;standalone&quot;,
  &quot;background_color&quot;: &quot;#ffffff&quot;,
  &quot;theme_color&quot;: &quot;#0e1117&quot;,
  &quot;icons&quot;: [
    {
      &quot;src&quot;: &quot;https://cdn-icons-png.flaticon.com/512/727/727799.png&quot;,
      &quot;sizes&quot;: &quot;512x512&quot;,
      &quot;type&quot;: &quot;image/png&quot;
    }
  ]
}">
<script>
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('data:text/javascript,self.addEventListener("fetch", function(e) {});');
  }
</script>
"""
st.markdown(pwa_code, unsafe_allow_html=True)

# --- TÍTULO E ABAS DO APLICATIVO ---
st.title("💧 Spray Drop Analyser — Análise de Papel Hidrossensível")

modo_app = st.sidebar.radio("🧭 Modo de Operação", ["📊 Análise Individual", "📂 Análise em Lote (Excel)"])

# --- BARRA LATERAL DE CONFIGURAÇÕES GLOBAIS ---
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parâmetros de Calibração")

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

canal_cor = st.sidebar.selectbox("Canal de Cor", ["Azul", "Verde", "Vermelho", "Cinza"], index=0)

sensibilidade_extra = st.sidebar.slider(
    "Ajuste Fino do Limiar (Offset)", 
    min_value=-50, 
    max_value=50, 
    value=0, 
    step=1
)

st.sidebar.subheader("📏 Limites de Classificação (µm)")
limite_pequena = st.sidebar.number_input("Limite Máx. p/ Gotas Pequenas", min_value=50, max_value=300, value=150, step=10)
limite_media = st.sidebar.number_input("Limite Máx. p/ Gotas Médias", min_value=100, max_value=600, value=250, step=10)

st.sidebar.subheader("🧹 Filtros de Ruído (Crivo e Teto)")
usar_crivo = st.sidebar.checkbox("Ativar Crivo Mínimo (Poeira)", value=False)
diametro_crivo_min = 20.0
if usar_crivo:
    diametro_crivo_min = st.sidebar.number_input("Descartar abaixo de (µm):", min_value=1.0, max_value=100.0, value=20.0, step=5.0)

usar_teto = st.sidebar.checkbox("Ativar Teto Máximo (Borrões)", value=True)
diametro_teto_max = 800.0
if usar_teto:
    diametro_teto_max = st.sidebar.number_input("Descartar acima de (µm):", min_value=300.0, max_value=2000.0, value=800.0, step=50.0)


# ==========================================
# MODO 1: ANÁLISE INDIVIDUAL
# ==========================================
if modo_app == "📊 Análise Individual":
    st.write("Faça o upload de uma foto individual para visualizar a binarização, contornos e o relatório detalhado.")
    
    uploaded_file = st.file_uploader("📂 Escolha a foto do papel hidrossensível", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
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

        # Limiar Otsu + Offset
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
        img_contornos = img_original.copy()

        for c in contornos:
            area_px = cv2.contourArea(c)
            if area_px > 0:
                area_um2 = area_px * area_por_pixel_um2
                diametro_um = 2 * math.sqrt(area_um2 / math.pi)

                if usar_crivo and diametro_um < diametro_crivo_min:
                    continue
                if usar_teto and diametro_um > diametro_teto_max:
                    continue

                soma_area_gotas_px += area_px
                total_gotas += 1

                volume_gota = (math.pi / 6.0) * (diametro_um ** 3)
                diametros_lista.append(diametro_um)
                volumes_lista.append(volume_gota)

                if diametro_um < limite_pequena:
                    pequenas += 1
                elif limite_pequena <= diametro_um <= limite_media:
                    medias += 1
                else:
                    grandes += 1

                cv2.drawContours(img_contornos, [c], -1, (0, 255, 0), 2)

        porcentagem_cobertura = (soma_area_gotas_px / total_pixels_img * 100.0) if total_pixels_img > 0 else 0.0

        # Cálculos Volumétricos ASABE
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

        # Layout de Exibição
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Imagem Original")
            st.image(image_pil, use_column_width=True)
        with col2:
            st.subheader("Gotas Detectadas")
            img_contornos_rgb = cv2.cvtColor(img_contornos, cv2.COLOR_BGR2RGB)
            st.image(img_contornos_rgb, use_column_width=True)

        st.markdown("---")
        st.subheader("📈 Relatório de Resultados")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total de Gotas", total_gotas)
        m2.metric("Cobertura", f"{porcentagem_cobertura:.2f}%")
        m3.metric("DMV (D0.5)", f"{dmv:.1f} µm")
        m4.metric("Span", f"{span:.2f}")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("D0.1", f"{d01:.1f} µm")
        r2.metric("D0.9", f"{d09:.1f} µm")
        r3.metric("Gotas Pequenas", pequenas)
        r4.metric("Gotas Médias/Grandes", medias + grandes)


# ==========================================
# MODO 2: ANÁLISE EM LOTE E EXPORTAÇÃO EXCEL
# ==========================================
else:
    st.write("📂 Faça o upload de **várias fotos** de uma vez para processar o lote completo e gerar a planilha consolidada.")
    
    uploaded_files = st.file_uploader(
        "Selecione as fotos do papel hidrossensível", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} imagem(ns) na fila de processamento.")
        
        if st.button("🚀 Processar Lote e Gerar Excel"):
            resultados_lote = []
            barra_progresso = st.progress(0)
            total_arquivos = len(uploaded_files)
            
            for i, uploaded_file in enumerate(uploaded_files):
                image_pil = Image.open(uploaded_file).convert('RGB')
                img_original = np.array(image_pil)
                img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)

                altura_px, largura_px, _ = img_original.shape
                total_pixels_img = largura_px * altura_px

                if canal_cor == 'Azul':
                    img_canal = img_original[:, :, 0]
                elif canal_cor == 'Verde':
                    img_canal = img_original[:, :, 1]
                elif canal_cor == 'Vermelho':
                    img_canal = img_original[:, :, 2]
                else:
                    img_canal = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

                limiar_otsu, _ = cv2.threshold(img_canal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                limiar_final = int(max(0, min(255, limiar_otsu + sensibilidade_extra)))
                _, img_binaria = cv2.threshold(img_canal, limiar_final, 255, cv2.THRESH_BINARY_INV)

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

                        if usar_crivo and diametro_um < diametro_crivo_min:
                            continue
                        if usar_teto and diametro_um > diametro_teto_max:
                            continue

                        soma_area_gotas_px += area_px
                        total_gotas += 1
                        volume_gota = (math.pi / 6.0) * (diametro_um ** 3)
                        diametros_lista.append(diametro_um)
                        volumes_lista.append(volume_gota)

                porcentagem_cobertura = (soma_area_gotas_px / total_pixels_img * 100.0) if total_pixels_img > 0 else 0.0

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

                resultados_lote.append({
                    "Arquivo": uploaded_file.name,
                    "Total Gotas": total_gotas,
                    "Cobertura (%)": round(porcentagem_cobertura, 2),
                    "D0.1 (µm)": round(d01, 1),
                    "DMV / D0.5 (µm)": round(dmv, 1),
                    "D0.9 (µm)": round(d09, 1),
                    "Span": round(span, 2)
                })

                barra_progresso.progress((i + 1) / total_arquivos)

            df_resultado = pd.DataFrame(resultados_lote)
            st.success("✅ Lote processado com sucesso!")
            st.dataframe(df_resultado)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_resultado.to_excel(writer, index=False, sheet_name='Relatorio_Lote')
            processed_data = output.getvalue()

            st.download_button(
                label="📥 Baixar Planilha Consolidada (.xlsx)",
                data=processed_data,
                file_name="relatorio_lote_gotas.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
