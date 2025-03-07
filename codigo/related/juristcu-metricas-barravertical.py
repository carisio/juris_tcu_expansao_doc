import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# Como mapas não são ordenados, define a ordem para a visualização 
# Obs.: nesse caso, a ordem default acaba coincidindo com o que queremos.
# Mas vamos deixar mais genérico
ordem_expansao_doc = ["BM25.dT5q",
                      "BM25.Syn(GPT3.5)", "BM25.Syn(GPT4o)", "BM25.Syn(Llama3)",
                      "BM25.dT5q.Syn(GPT35)", "BM25.dT5q.Syn(GPT4o)", "BM25.dT5q.Syn(Llama3)"]
ordem_semantico = ["BERT.pt.TCU", "BERT.pt.base", "BERT.pt.legal", "BERT.pt.STJ",
                   "BERT.ml",
                   "OpenAI.small", "OpenAI.large"]

# Define as métricas para os grupos G1, G2 e G3
metricas_g1 = {
    "Baseline"                 : [23.8, 19.7, 53.9, 27.5],
    "expansao_doc": {
        "BM25.dT5q"            : [33.6, 27.4, 65.4, 38.5],
        "BM25.Syn(GPT3.5)"     : [26.8, 22.1, 55.1, 30.1],
        "BM25.Syn(GPT4o)"      : [25.8, 21.4, 55.6, 29.7],
        "BM25.Syn(Llama3)"     : [27.4, 22.7, 54.8, 30.8],
        "BM25.dT5q.Syn(GPT35)" : [35.4, 28.8, 67.5, 40.4],
        "BM25.dT5q.Syn(GPT4o)" : [34.6, 28.2, 65.8, 39.2],
        "BM25.dT5q.Syn(Llama3)": [35.2, 28.7, 69.1, 40.2]
        },
    "semantico": {
        "BERT.pt.TCU"          : [4.4 , 3.5 , 17.4, 5.6 ],
        "BERT.pt.base"         : [7.4 , 6.3 , 19.9, 8.3 ],
        "BERT.pt.legal"        : [10.8, 8.8 , 28.8, 12.1],
        "BERT.pt.STJ"          : [20.2, 16.3, 42.4, 22.6],
        "BERT.ml"              : [14.6, 11.6, 35.1, 16.1],
        "OpenAI.small"         : [37.8, 30.7, 74.9, 44.5],
        "OpenAI.large"         : [40.8, 33.2, 75.4, 47.3]
    }
}

metricas_g2 = {
    "Baseline"                 : [37.8, 31.8, 86.7, 51.1],
    "expansao_doc": {
        "BM25.dT5q"            : [40.2, 33.8, 88.2, 54.6],
        "BM25.Syn(GPT3.5)"     : [40.8, 34.3, 88.5, 54.4],
        "BM25.Syn(GPT4o)"      : [39.4, 33.0, 90.5, 53.6],
        "BM25.Syn(Llama3)"     : [39.4, 33.1, 87.4, 53.0],
        "BM25.dT5q.Syn(GPT35)" : [42.4, 35.6, 89.3, 56.6],
        "BM25.dT5q.Syn(GPT4o)" : [41.8, 35.0, 89.7, 56.3],
        "BM25.dT5q.Syn(Llama3)": [42.0, 35.3, 90.9, 56.5]
        },
    "semantico": {
        "BERT.pt.TCU"          : [11.0, 9.2 , 35.9, 13.4],
        "BERT.pt.base"         : [15.6, 13.2, 41.8, 18.7],
        "BERT.pt.legal"        : [16.6, 13.7, 46.4, 21.0],
        "BERT.pt.STJ"          : [30.2, 24.8, 71.3, 38.5],
        "BERT.ml"              : [24.6, 20.3, 60.5, 31.5],
        "OpenAI.small"         : [46.8, 38.9, 89.5, 58.8],
        "OpenAI.large"         : [49.2, 40.8, 89.2, 61.8]
    }
}

metricas_g3 = {
    "Baseline"                 : [38.8, 34.5, 91.8, 53.3],
    "expansao_doc": {
        "BM25.dT5q"            : [40.8, 36.2, 93.9, 55.6],
        "BM25.Syn(GPT3.5)"     : [40.6, 36.1, 91.5, 54.6],
        "BM25.Syn(GPT4o)"      : [40.8, 36.3, 93.4, 55.2],
        "BM25.Syn(Llama3)"     : [39.6, 35.2, 92.3, 54.1],
        "BM25.dT5q.Syn(GPT35)" : [41.6, 36.9, 91.9, 55.7],
        "BM25.dT5q.Syn(GPT4o)" : [41.6, 36.9, 94.0, 56.4],
        "BM25.dT5q.Syn(Llama3)": [42.0, 37.2, 92.9, 56.4]
        },
    "semantico": {
        "BERT.pt.TCU"          : [20.2, 18.0, 60.8, 28.8],
        "BERT.pt.base"         : [22.2, 19.6, 60.7, 28.9],
        "BERT.pt.legal"        : [18.2, 16.1, 49.2, 23.4],
        "BERT.pt.STJ"          : [34.8, 30.7, 86.8, 46.0],
        "BERT.ml"              : [34.4, 30.5, 79.2, 45.2],
        "OpenAI.small"         : [48.2, 42.5, 91.7, 60.9],
        "OpenAI.large"         : [47.2, 41.5, 91.5, 60.8]
    }
}


# Cálculo do ganho/perda percentual em relação ao baseline
def percentual_do_baseline(resultados, baseline):
    if isinstance(resultados, (int, float)):
        return (resultados - baseline) / baseline * 100
    
    return [(v - baseline) / baseline * 100 for v in resultados]


def plot_metricas(grupo, calcular_percentual=True, salvar_figura=False, baseline_no_topo=True):
    if grupo == 1:
        metricas = metricas_g1
    elif grupo == 2:
        metricas = metricas_g2
    elif grupo == 3:
        metricas = metricas_g3

    # Configuração dos subplots
    label_metricas = ["P@10", "R@10", "MRR@10", "nDCG@10"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Query group: G{grupo}")

    mapa = plt.cm.get_cmap("tab10", 20)
    #colors = {"BM25 Variants": colors(0), "BERT Variants": colors(1)}
    #0055FFFF, #3399FFFF, #66CCFFFF, #99EEFFFF, #CCFFFFFF, #FFFFCCFF, #FFEE99FF, #FFCC66FF, #FF9933FF, #FF5500FF
    colors = {
        "baseline": "lightgray",
        "expansao_doc": "lightgreen",
        "semantico": "lightsalmon"
    }
    alpha=0.7

    for idx_metrica, label_metrica in enumerate(label_metricas):
        ax = axes[idx_metrica // 2, idx_metrica % 2]
        
        # Dados da métrica atual. Baseline já é um vetor
        # expansao_doc e semantico são mapas. Precisamos extrair os vetores
        # do mapa na ordem em que devem ser visualizados
        # idx_metrica é de 0 a 3 e representa P, R, MRR e nDCG
        baseline_value = metricas["Baseline"][idx_metrica]
        expansao_doc_values = [metricas["expansao_doc"][metodo][idx_metrica] for metodo in ordem_expansao_doc]
        semantico_values = [metricas["semantico"][metodo][idx_metrica] for metodo in ordem_semantico]
        
        # Converte para percentual
        if calcular_percentual:
            expansao_doc_values = percentual_do_baseline(expansao_doc_values, baseline_value)
            semantico_values = percentual_do_baseline(semantico_values, baseline_value)
  
        # Índices para as barras
        sep_grupos = 1
        y_expansao_doc = np.arange(len(ordem_expansao_doc))
        y_semantico = np.arange(len(ordem_semantico)) + len(ordem_expansao_doc) + sep_grupos  # Separação entre os grupos
        if not calcular_percentual: # Se não for percentual, aí tem que adicionar o baseline
            y_baseline = [0]
            y_expansao_doc = y_expansao_doc + len(y_baseline) + sep_grupos
            y_semantico = y_semantico + len(y_baseline) + sep_grupos
    
        # Barras
        if not calcular_percentual: # Se não for percentual, aí tem que adicionar o baseline
            ax.barh(y=y_baseline, width=[baseline_value], color=colors["baseline"], label="baseline")
        barras_exp_doc = ax.barh(y=y_expansao_doc, width=expansao_doc_values, color=colors["expansao_doc"], label="expansao_doc", alpha=alpha)
        barras_semantico = ax.barh(y=y_semantico, width=semantico_values, color=colors["semantico"], label="semantico", alpha=alpha)

        # Pra inverter, tem que corrigir a posição do label depois no ax.text
        if baseline_no_topo:
            ax.invert_yaxis()
        
        # Rótulos percentual em cima da barra
        if not calcular_percentual:
            for barra in barras_exp_doc + barras_semantico:
                altura = barra.get_width()
                valor_para_exibir = round(percentual_do_baseline(altura, baseline_value))
                posicao_x_para_exibir = barra.get_width() + 6 if valor_para_exibir < -50 else barra.get_width() / 2
                valor_para_exibir = f"{'+' if valor_para_exibir > 0 else ''}{valor_para_exibir}%"
                ax.text(posicao_x_para_exibir,
                        barra.get_y() + (barra.get_height() if baseline_no_topo else 0),  # Posição do texto
                        valor_para_exibir,  # O texto a ser exibido
                        ha='center', va='bottom', fontsize=12)  # Alinhamento e tamanho da fonte
            ax.set_xlim(0, 100)
    
        # Rótulos no eixo x e y e título
        if calcular_percentual:
            # Eixo x
            ax.set_yticks(list(y_expansao_doc) + list(y_semantico))
            ax.set_yticklabels(ordem_expansao_doc + ordem_semantico)
            # Eixo y
            ax.set_xlabel("Percentage relative to the baseline")
            # Título
            ax.set_title(label_metrica)
        else:
            # Eixo x
            ax.set_yticks(list(y_baseline) + list(y_expansao_doc) + list(y_semantico))
            ax.set_yticklabels(["Baseline"] + ordem_expansao_doc + ordem_semantico)
            # Eixo y
            ax.set_xlabel(f"{label_metrica}")
            # Título
            #ax.set_title(label_metrica)
        
        
        # Linha de referência no 0 %
        if calcular_percentual:
            ax.axvline(0, color="black", linewidth=0.8)
        else:
            ax.axvline(baseline_value, color="black", linestyle="--", linewidth=0.8)
                
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
               

    # Criando os elementos da legenda com as cores corretas
    baseline_patch = mpatches.Patch(color=colors['baseline'], label="Baseline")
    expansao_doc_patch = mpatches.Patch(color=colors['expansao_doc'], label="Document expansion variants", alpha=alpha)
    semantico_patch = mpatches.Patch(color=colors['semantico'], label="Semantic variants", alpha=alpha)

    # Criando a legenda global
    if calcular_percentual:
        fig.legend(handles=[expansao_doc_patch, semantico_patch],
               loc="lower center",
               bbox_to_anchor=(0.55, 0.45))
    else:
        fig.legend(handles=[baseline_patch, expansao_doc_patch, semantico_patch],
               loc="lower center",
               bbox_to_anchor=(0.55, 0.45))
    
    plt.tight_layout(rect=[0, 0, 1, 1], h_pad=5, w_pad=5)
    
    if salvar_figura:
        plt.savefig(f"metrics_G{grupo}.png", dpi=600, bbox_inches='tight')
    plt.show()
        

salvar_figura=True
baseline_no_topo=True
plot_metricas(grupo=1, calcular_percentual=False, salvar_figura=salvar_figura, baseline_no_topo=baseline_no_topo)
plot_metricas(grupo=3, calcular_percentual=False, salvar_figura=salvar_figura, baseline_no_topo=baseline_no_topo)
plot_metricas(grupo=2, calcular_percentual=False, salvar_figura=salvar_figura, baseline_no_topo=baseline_no_topo)