import numpy as np
from scipy import stats
from scipy.optimize import least_squares

# ==========================================
# 1. SOLUCIONADOR NUMÉRICO DO MÉTODO DOS MOMENTOS
# ==========================================
def estimar_por_momentos(dados, modelo_distribuicao, chute_inicial, fixar_loc_zero=True):
    """
    Estima os parâmetros de qualquer distribuição usando o Método dos Momentos Numérico.
    """
    dados = np.array(dados)
    dist = getattr(stats, modelo_distribuicao)
    num_params_estimados = len(chute_inicial)
    
    # 1. Calcula os Momentos Amostrais (Reais)
    # Ex: Para 2 parâmetros, calcula a Média (k=1) e a Esperança de X² (k=2)
    momentos_amostrais = [np.mean(dados**k) for k in range(1, num_params_estimados + 1)]
    
    # 2. Função Objetivo para o Otimizador
    def funcao_erro(parametros_tentativa):
        # Monta a lista de argumentos. A maioria dos modelos em confiabilidade fixa a localização em 0.
        if fixar_loc_zero:
            # Ex: Para Weibull (shape, scale), o SciPy exige (shape, loc, scale)
            args = list(parametros_tentativa[:-1]) + [0] + [parametros_tentativa[-1]]
        else:
            args = parametros_tentativa
            
        erros = []
        for k in range(1, num_params_estimados + 1):
            # Pede ao SciPy o momento teórico da distribuição com esses parâmetros
            momento_teorico = dist.moment(k, *args)
            
            # Usamos o erro relativo [(Teórico - Real) / Real] para evitar que momentos 
            # de alta ordem (como X³) dominem o otimizador por serem números muito grandes.
            erro_relativo = (momento_teorico - momentos_amostrais[k-1]) / momentos_amostrais[k-1]
            erros.append(erro_relativo)
            
        return erros

    # 3. Executa a Otimização (Busca o erro zero)
    resultado = least_squares(funcao_erro, chute_inicial, method='lm')
    
    # 4. Organiza a saída
    parametros_estimados = resultado.x
    if fixar_loc_zero:
        parametros_finais = tuple(list(parametros_estimados[:-1]) + [0.0] + [parametros_estimados[-1]])
    else:
        parametros_finais = tuple(parametros_estimados)
        
    return parametros_finais, momentos_amostrais, resultado.success

# ==========================================
# 2. INTERFACE DO USUÁRIO
# ==========================================
print("\n=== ESTIMADOR UNIVERSAL: MÉTODO DOS MOMENTOS ===")
try:
    entrada_dados = input("Digite a amostra de dados separados por ESPAÇO: ").replace(',', '.')
    amostra = [float(v) for v in entrada_dados.split()]
    
    # Define o modelo
    print("\nModelos disponíveis:")
    print("- 'norm'           (Normal/Gaussiana)")
    print("- 'expon'          (Exponencial)")
    print("- 'uniform'        (Uniforme)")
    print("- 'gamma'          (Gama)")
    print("- 'beta'           (Beta)")
    print("- 'weibull_min'    (Weibull Mínimo)")
    print("- 'weibull_max'    (Weibull Máximo)")
    print("- 'lognorm'        (Log-Normal)")
    print("- 'chi2'           (Qui-Quadrado)")
    print("- 't'              (t de Student)")
    print("- 'f'              (F de Snedecor)")
    print("- 'laplace'        (Laplace)")
    print("- 'cauchy'         (Cauchy)")
    print("- 'poisson'        (Poisson)")
    print("- 'binom'          (Binomial - Quantos sucessos em n tentativas?)")
    print("- 'nbinom'         (Binomial - Quantas tentativas até n sucessos?)")
    modelo = input("Digite a sigla do modelo: ").strip().lower()

    print("\n[Localização]")
    print("Se for rodar distribuições gerais como Normal (norm), Uniforme ou Laplace, não fixe a ")
    print("localização no zero (responda n). A loc é a própria média ou início da curva.")
    print("Para distribuições de tempo de vida (Exponencial, Weibull, Gama, Log-Normal), ou")
    print("distribuições discretas (Poisson, Binomial), fixe no zero (responda s).")
    
    loc_input = input("\nFixar a localização (loc) no zero? Recomendado para confiabilidade (s/n): ").strip().lower()
    fixar_loc = True if loc_input == 's' else False
    
    print("\n[Chute Inicial]")
    print("O otimizador precisa de uma estimativa para começar a busca (use médias amostrais simples).")
    print("Se você FIXOU a localização (s):")
    print("- 'expon' / 'poisson'        : Digite 1 valor  (ex: escala ou média).")
    print("- 'weibull_min' / '_max'     : Digite 2 valores (ex: forma, escala).")
    print("- 'gamma' / 'lognorm'        : Digite 2 valores (ex: forma, escala).")
    print("- 'chi2' / 'binom' / 'nbinom': Digite 2 valores (ex: graus_liberdade/n, escala/prob_p).")
    print("- 'beta' / 'f'               : Digite 3 valores (ex: forma1, forma2, escala).")
    
    print("\nSe você NÃO FIXOU a localização (n):")
    print("- 'norm' / 'laplace' / 'cauchy': Digite 2 valores (ex: média, desvio/escala).")
    print("- 'uniform'                  : Digite 2 valores (ex: início, amplitude).")
    print("- 't'                        : Digite 3 valores (ex: graus_liberdade, média, escala).")
    
    # Dica extra de segurança para o usuário
    if fixar_loc:
        print("\n* Como você fixou a loc em 0, não digite a loc no chute inicial!")
    else:
        print("\n* Como você NÃO fixou a loc, lembre-se de que um dos valores do chute deve ser a própria loc.")
        
    entrada_chute = input("\nDigite o chute inicial separado por espaço: ").replace(',', '.')
    chute = [float(v) for v in entrada_chute.split()]
    
    print(f"\nCalculando parâmetros para '{modelo}' via Método dos Momentos...")
    
    params_finais, m_amostrais, sucesso = estimar_por_momentos(amostra, modelo, chute, fixar_loc)
    
    print("-" * 60)
    print("MOMENTOS DA AMOSTRA:")
    for i, m in enumerate(m_amostrais, 1):
        print(f"  Momento {i} (m_{i}): {m:.4f}")
        
    print("\nRESULTADO DA ESTIMAÇÃO:")
    if sucesso:
        print(f"  Parâmetros Brutos (SciPy): {tuple(round(p, 4) for p in params_finais)}")
        
        # --- TRADUTOR PARA ENGENHARIA DE CONFIABILIDADE ---
        print("\n  [TRADUÇÃO DOS PARÂMETROS PARA ENGENHARIA]")
        if modelo == 'expon':
            escala = params_finais[1]
            lamb = 1.0 / escala if escala != 0 else 0
            print(f"  -> Taxa de Falha (\u03bb): {lamb:.6f}")
            print(f"  -> MTTF (\u03b8):         {escala:.4f}")
            
        elif modelo == 'weibull_min':
            print(f"  -> Parâmetro de Forma (\u03b2): {params_finais[0]:.4f}")
            print(f"  -> Parâmetro de Escala (\u03b7): {params_finais[2]:.4f}")
            
        elif modelo == 'norm':
            print(f"  -> Média (\u03bc):          {params_finais[0]:.4f}")
            print(f"  -> Desvio Padrão (\u03c3): {params_finais[1]:.4f}")
            
        elif modelo == 'gamma':
            forma = params_finais[0]
            escala = params_finais[2]
            print(f"  -> Forma (k ou \u03b1):      {forma:.4f}")
            print(f"  -> Escala (\u03b8):         {escala:.4f}")
            print(f"  -> Taxa (\u03bb = 1/\u03b8):     {1.0/escala:.6f}")
    else:
        print("  [!] O otimizador não conseguiu convergir.")
    print("-" * 60)

except Exception as e:
    print(f"\nOcorreu um erro: {e}")