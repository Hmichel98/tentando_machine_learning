import numpy as np
import time


def regressao_multilinear(X: np.ndarray, y: np.ndarray) -> type(lambda x_1, x_2: m1 * x_1 + m2 * x_2 + n1):
    """
    Regressão linear com multivariáveis independentes utilizando:
    - Gradient Descent: algorítimo iterável para se achar um mínimo local da função;
    - Closure: uma função externa que retorna uma função interna que utiliza variáveis do enclosing scope.
    """

    # Quantidade de iterações (obs: outro método de parada é a precisão)
    epochs = 100000
    # Controla a largura da passada na atualização dos pesos
    gamma = 0.0001

    # Os pesos das variáveis independentes
    coef_angs = np.random.randn(X.shape[1],) 
    # Coeficiente Linear
    coef_linear = 0

    # constante presente nas derivadas parciais
    # Evita cálculo repetido (uma espécie de 'colocar em evidência')
    constante = (-2/X.shape[1])

    for _ in range(epochs):
        # predição do modelo que será comparado utilizando Mean Squared Error
        y_modelo = np.dot(X, coef_angs) + coef_linear

        # Fator repetido nos dois cálculos; 
        # Colocamos em evidência
        diferenca = y - y_modelo

        # Derivadas parcias de todos os pesos das variáveis independentes
        dl_dms = constante * np.dot(X.transpose(), diferenca)

        # Derivada parcial em função do coeficiente linear
        dl_dn = constante * np.sum(diferenca) 

        # Atualização dos coeficientes utilizando Gradient Descent
        coef_angs -= gamma * dl_dms
        coef_linear -= gamma * dl_dn

    
    def predict(new_X: np.ndarray) -> float:
        """Função que retorna a predição do modelo multilinear"""
        return np.dot(new_X, coef_angs) + coef_linear

    # Retornando a função que será utilizado para novas previsões
    return predict


if __name__ == "__main__":
    from sklearn.datasets import load_boston
    boston = load_boston()
    x = boston.data
    y = boston.target


    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler()
    x = sc.fit_transform(x)

    predict = regressao_multilinear(x, y)

    print(predict.__closure__[0].cell_contents)
    print(predict.__closure__[1].cell_contents)
