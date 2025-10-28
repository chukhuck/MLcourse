from tensorflow.keras import backend, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def build_and_train_model(X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    y_pred = model.predict(X_test).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mae, mse, r2

def apply_pca_analysis(X_train, X_test, variance_threshold=0.95):
    pca = PCA()
    pca.fit(X_train)
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    print(f"Рекомендуемое количество компонент: {n_components}")
    print(f"Объясненная дисперсия: {cumulative_variance[n_components-1]:.3f}")
    
    # Применяем PCA с выбранным количеством компонент
    pca_reduced = PCA(n_components=n_components)
    X_train_pca = pca_reduced.fit_transform(X_train)
    X_test_pca = pca_reduced.transform(X_test)
    
    return X_train_pca, X_test_pca, pca_reduced

def apply_pca_analysis_with_graphic(X_train, X_test, variance_threshold=0.95):
    """
    Анализирует данные с помощью PCA и возвращает преобразованные данные
    """
    pca = PCA()
    pca.fit(X_train)
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    print(f"Рекомендуемое количество компонент: {n_components}")
    print(f"Объясненная дисперсия: {cumulative_variance[n_components-1]:.3f}")
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance, 'b-', linewidth=2)
    plt.axhline(y=variance_threshold, color='r', linestyle='--', alpha=0.7, label=f'{variance_threshold*100}% дисперсии')
    plt.axvline(x=n_components-1, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Количество главных компонент')
    plt.ylabel('Накопленная доля объясненной дисперсии')
    plt.title('Кумулятивная дисперсия PCA')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Применяем PCA с выбранным количеством компонент
    pca_reduced = PCA(n_components=n_components)
    X_train_pca = pca_reduced.fit_transform(X_train)
    X_test_pca = pca_reduced.transform(X_test)
    
    return X_train_pca, X_test_pca, pca_reduced

def empty_function():
    pass

