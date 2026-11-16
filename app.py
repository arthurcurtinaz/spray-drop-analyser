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

# --- FUNÇÃO AUXILIAR DE PROCESSAMENTO ---
def processar_imagem_unica(img_bgr, limiar_aplicado, area_fisica_cm2, usar_crivo, diametro_crivo_min, usar_teto, diametro_teto_max, limite_pequena, limite_grande, desenhar_contornos=False):
    altura_px, largura_px, _ = img_bgr.shape
    total_pixels_img = largura_px * altura_px

    # Canal Verde fixo (índice 1 no padrão BGR do OpenCV)
    img_canal = img_bgr[:, :, 1]
    _, img_binaria = cv2.threshold(img_canal, limiar_aplicado, 255, cv2.THRESH_BINARY_INV)

    area_total_um2 = area_fisica_cm2 * 100000000.0
    area_por_pixel_um2 = area_total_um2 / total_pixels_img if total_pixels_img > 0 else 1.0

    contornos, _ = cv2.findContours(img_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_gotas = 0
    gotas_finas = 0
    gotas_medias = 0
    gotas_grossas = 0
    soma_area_gotas_px = 0
    
    diametros_lista = []
    volumes_lista = []
    img_contornos = img_bgr.copy() if desenhar_contornos else None

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
                gotas_finas += 1
            elif diametro_um > limite_grande:
                gotas_grossas += 1
            else:
                gotas_medias += 1

            if desenhar_contornos:
                cv2.drawContours(img_contornos, [c], -1, (0, 255, 0), 2)

    porcentagem_cobertura = (soma_area_gotas_px / total_pixels_img * 100.0) if total_pixels_img > 0 else 0.0

    d01, dmv, d09, span = 0.0, 0.0, 0.0, 0.0
    pct_vol_deriva = 0.0
    pct_vol_escorrimento = 0.0

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

            # Cálculo % Volume em gotas < 150 µm (Potencial de Deriva)
            vol_deriva = np.sum(v_arr[d_arr < 150.0])
            pct_vol_deriva = (vol_deriva / vol_total) * 100.0

            # Cálculo % Volume em gotas > 600 µm (Potencial de Escorrimento)
            vol_escorrimento = np.sum(v_arr[d_arr > 600.0])
            pct_vol_escorrimento = (vol_escorrimento / vol_total) * 100.0

    return {
        "total_gotas": total_gotas,
        "cobertura": porcentagem_cobertura,
        "finas": gotas_finas,
        "medias": gotas_medias,
        "grossas": gotas_grossas,
        "d01": d01,
        "dmv": dmv,
        "d09": d09,
        "span": span,
        "pct_vol_deriva": pct_vol_deriva,
        "pct_vol_escorrimento": pct_vol_escorrimento,
        "img_contornos": img_contornos
    }


# --- TÍTULO E BARRA LATERAL DO APLICATIVO ---
st.title("💧 Spray Drop Analyser — Análise de Papel Hidrossensível")

modo_app = st.sidebar.radio("🧭 Modo de Operação", ["📊 Análise Individual", "📂 Análise em Lote (Excel)"])

metodo_otsu_lote = None
if modo_app == "📂 Análise em Lote (Excel)":
    metodo_otsu_lote = st.sidebar.radio(
        "🎯 Limiar de Otsu no Lote:",
        [
            "Individual por foto",
            "Global unificado (mesmo p/ todas)"
        ],
        help="Individual: calcula um Otsu para cada imagem. Global: calcula a mediana de Otsu do lote inteiro e aplica o mesmo corte a todas as fotos."
    )

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

sensibilidade_extra = st.sidebar.slider(
    "Ajuste Fino do Limiar (Offset)", 
    min_value=-50, 
    max_value=50, 
    value=0, 
    step=1
)

st.sidebar.subheader("📏 Limites de Classificação (µm)")
limite_pequena = st.sidebar.number_input("Limite Máx. p/ Gotas Finas", min_value=50, max_value=300, value=150, step=10)
limite_grande = st.sidebar.number_input("Limite Mín. p/ Gotas Médias/Grossas", min_value=200, max_value=600, value=300, step=10)

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
    st.write("Faça o upload de uma foto individual para visualizar o limiar calculado (Canal Verde), binarização, contornos e o relatório detalhado.")
    
    uploaded_file = st.file_uploader("📂 Escolha a foto do papel hidrossensível", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert('RGB')
        img_original = np.array(image_pil)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)

        img_canal = img_original[:, :, 1]
        limiar_otsu, _ = cv2.threshold(img_canal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        limiar_final = int(max(0, min(255, limiar_otsu + sensibilidade_extra)))

        res = processar_imagem_unica(
            img_original, limiar_final, area_fisica_cm2,
            usar_crivo, diametro_crivo_min, usar_teto, diametro_teto_max,
            limite_pequena, limite_grande, desenhar_contornos=True
        )

        # Layout de Exibição de Imagens
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Imagem Original")
            st.image(image_pil, use_container_width=True)
        with col2:
            st.subheader("Gotas Detectadas")
            img_contornos_rgb = cv2.cvtColor(res["img_contornos"], cv2.COLOR_BGR2RGB)
            st.image(img_contornos_rgb, use_container_width=True)

        st.markdown("---")
        
        st.info(f"🔍 **Parâmetros de Binarização (Canal Verde):** Limiar base de Otsu = **{int(limiar_otsu)}** | Limiar Aplicado (com Offset) = **{limiar_final}**")

        st.subheader("📈 Relatório de Resultados")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total de Gotas", res["total_gotas"])
        m2.metric("Cobertura", f"{res['cobertura']:.2f}%")
        m3.metric("DMV (D0.5)", f"{res['dmv']:.1f} µm")
        m4.metric("Span", f"{res['span']:.2f}")

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("D0.1", f"{res['d01']:.1f} µm")
        r2.metric("D0.9", f"{res['d09']:.1f} µm")
        r3.metric(f"Finas (<{limite_pequena}µm)", res["finas"])
        r4.metric("Médias", res["medias"])
        r5.metric(f"Grossas (>{limite_grande}µm)", res["grossas"])

        st.markdown("### 📊 Indicador de Homogeneidade (Span)")
        span_proporcao = min(max(res["span"] / 2.0, 0.0), 1.0)
        st.progress(span_proporcao)
        if res["span"] < 1.0:
            st.success("✅ **Homogeneidade Excelente:** Span baixo (< 1.0) indica gotas de tamanhos muito uniformes.")
        elif res["span"] <= 1.5:
            st.warning("⚠️ **Homogeneidade Moderada:** Span entre 1.0 e 1.5.")
        else:
            st.error("❌ **Homogeneidade Baixa:** Span alto (> 1.5) indica alta dispersão de tamanhos de gotas.")

        # --- QUADRO SEPARADO: RISCOS DE PERDA DE CALDA (DERIVA E ESCORRIMENTO) ---
        st.markdown("---")
        with st.container(border=True):
            st.subheader("⚠️ Análise Volumétrica de Risco de Perdas")
            p1, p2 = st.columns(2)
            
            with p1:
                st.metric(
                    label="💨 Volume em Gotas < 150 µm (Potencial de Deriva)",
                    value=f"{res['pct_vol_deriva']:.2f}%",
                    help="Porcentagem do volume total de calda concentrada em gotas menores que 150 µm, suscetíveis ao arraste pelo vento ou evaporação."
                )
                if res['pct_vol_deriva'] > 15.0:
                    st.caption("🔴 Atenção: Alto percentual de volume sujeito à deriva (> 15%).")
                else:
                    st.caption("🟢 Nível de deriva dentro de limites aceitáveis.")
                    
            with p2:
                st.metric(
                    label="🌊 Volume em Gotas > 600 µm (Potencial de Escorrimento)",
                    value=f"{res['pct_vol_escorrimento']:.2f}%",
                    help="Porcentagem do volume total de calda concentrada em gotas maiores que 600 µm, que tendem a coalescer e escorrer da folha."
                )
                if res['pct_vol_escorrimento'] > 15.0:
                    st.caption("🔴 Atenção: Alto percentual de volume sujeito a escorrimento (> 15%).")
                else:
                    st.caption("🟢 Risco de escorrimento controlado.")


# ==========================================
# MODO 2: ANÁLISE EM LOTE E EXPORTAÇÃO EXCEL
# ==========================================
else:
    st.write("📂 Faça o upload de **várias fotos** de uma vez para processar o lote completo e gerar a planilha detalhada em Excel.")
    
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
            
            imagens_carregadas = []
            otsus_individuais = []
            
            for uploaded_file in uploaded_files:
                image_pil = Image.open(uploaded_file).convert('RGB')
                img_original = np.array(image_pil)
                img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
                
                img_canal = img_original[:, :, 1]
                limiar_otsu, _ = cv2.threshold(img_canal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                imagens_carregadas.append((uploaded_file.name, img_original, limiar_otsu))
                otsus_individuais.append(limiar_otsu)
            
            usar_global = "Global" in metodo_otsu_lote
            otsu_global_calculado = int(np.median(otsus_individuais)) if usar_global else None
            
            if usar_global:
                limiar_global_final = int(max(0, min(255, otsu_global_calculado + sensibilidade_extra)))
                st.info(f"🌐 **Limiar Global Unificado Calculado:** Otsu Mediano = **{otsu_global_calculado}** | Aplicado a todas (com Offset) = **{limiar_global_final}**")

            for i, (nome_arquivo, img_original, otsu_ind) in enumerate(imagens_carregadas):
                if usar_global:
                    otsu_base_exibir = otsu_global_calculado
                    limiar_final = limiar_global_final
                else:
                    otsu_base_exibir = int(otsu_ind)
                    limiar_final = int(max(0, min(255, otsu_ind + sensibilidade_extra)))

                res = processar_imagem_unica(
                    img_original, limiar_final, area_fisica_cm2,
                    usar_crivo, diametro_crivo_min, usar_teto, diametro_teto_max,
                    limite_pequena, limite_grande, desenhar_contornos=False
                )

                resultados_lote.append({
                    "Arquivo": nome_arquivo,
                    "Modo Limiar": "Global" if usar_global else "Individual",
                    "Limiar Otsu": otsu_base_exibir,
                    "Limiar Usado": limiar_final,
                    "Total Gotas": res["total_gotas"],
                    "Cobertura (%)": round(res["cobertura"], 2),
                    f"Gotas Finas (<{limite_pequena}µm)": res["finas"],
                    "Gotas Médias": res["medias"],
                    f"Gotas Grossas (>{limite_grande}µm)": res["grossas"],
                    "D0.1 (µm)": round(res["d01"], 1),
                    "DMV / D0.5 (µm)": round(res["dmv"], 1),
                    "D0.9 (µm)": round(res["d09"], 1),
                    "Span": round(res["span"], 2),
                    "Vol. < 150µm - Deriva (%)": round(res["pct_vol_deriva"], 2),
                    "Vol. > 600µm - Escorrimento (%)": round(res["pct_vol_escorrimento"], 2)
                })

                barra_progresso.progress((i + 1) / total_arquivos)

            df_resultado = pd.DataFrame(resultados_lote)
            st.success("✅ Lote processado com sucesso!")
            st.dataframe(df_resultado, use_container_width=True)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_resultado.to_excel(writer, index=False, sheet_name='Relatorio_Lote')
            processed_data = output.getvalue()

            st.download_button(
                label="📥 Baixar Planilha Consolidada Completa (.xlsx)",
                data=processed_data,
                file_name="relatorio_lote_gotas_completo.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
