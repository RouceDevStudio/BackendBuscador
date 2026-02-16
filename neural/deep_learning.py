"""
NEXUS NEURAL NETWORK - Sistema de 50+ Neuronas
Motor de IA avanzado para búsqueda inteligente
"""

import numpy as np
import json
import pickle
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple
import hashlib

class NeuralLayer:
    """Capa neuronal individual"""
    def __init__(self, input_size: int, output_size: int, activation='relu'):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.activation = activation
        
        # Para backpropagation
        self.input_cache = None
        self.output_cache = None
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, inputs):
        """Propagación hacia adelante"""
        self.input_cache = inputs
        z = np.dot(inputs, self.weights) + self.bias
        
        if self.activation == 'relu':
            self.output_cache = self.relu(z)
        elif self.activation == 'sigmoid':
            self.output_cache = self.sigmoid(z)
        elif self.activation == 'tanh':
            self.output_cache = self.tanh(z)
        elif self.activation == 'softmax':
            self.output_cache = self.softmax(z)
        else:
            self.output_cache = z
            
        return self.output_cache
    
    def backward(self, gradient, learning_rate=0.001):
        """Retropropagación"""
        if self.activation == 'relu':
            gradient = gradient * self.relu_derivative(self.output_cache)
        elif self.activation == 'sigmoid':
            gradient = gradient * self.sigmoid_derivative(self.input_cache @ self.weights + self.bias)
        elif self.activation == 'tanh':
            gradient = gradient * self.tanh_derivative(self.input_cache @ self.weights + self.bias)
        
        # Actualizar pesos y bias
        self.weights -= learning_rate * (self.input_cache.T @ gradient)
        self.bias -= learning_rate * np.sum(gradient, axis=0, keepdims=True)
        
        # Retornar gradiente para la capa anterior
        return gradient @ self.weights.T


class DeepNeuralNetwork:
    """Red neuronal profunda con 50+ capas configurables"""
    
    def __init__(self, architecture: List[int], activations: List[str] = None):
        """
        Args:
            architecture: Lista con el tamaño de cada capa [input, hidden1, hidden2, ..., output]
            activations: Lista de funciones de activación para cada capa
        """
        self.architecture = architecture
        self.layers = []
        
        if activations is None:
            activations = ['relu'] * (len(architecture) - 2) + ['softmax']
        
        # Crear capas
        for i in range(len(architecture) - 1):
            layer = NeuralLayer(
                architecture[i], 
                architecture[i + 1],
                activations[i]
            )
            self.layers.append(layer)
        
        self.training_history = []
        
    def forward(self, X):
        """Propagación hacia adelante a través de todas las capas"""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, gradient, learning_rate=0.001):
        """Retropropagación a través de todas las capas"""
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)
    
    def train(self, X, y, epochs=100, batch_size=32, learning_rate=0.001, verbose=True):
        """Entrenar la red neuronal"""
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Mezclar datos
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            total_loss = 0
            
            # Mini-batches
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]
                
                # Forward pass
                predictions = self.forward(batch_X)
                
                # Calcular pérdida (cross-entropy)
                loss = -np.mean(batch_y * np.log(predictions + 1e-8))
                total_loss += loss
                
                # Backward pass
                gradient = predictions - batch_y
                self.backward(gradient, learning_rate)
            
            avg_loss = total_loss / (n_samples / batch_size)
            self.training_history.append(avg_loss)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")
    
    def predict(self, X):
        """Hacer predicciones"""
        return self.forward(X)
    
    def save(self, filepath):
        """Guardar el modelo"""
        model_data = {
            'architecture': self.architecture,
            'layers': [],
            'training_history': self.training_history
        }
        
        for layer in self.layers:
            model_data['layers'].append({
                'weights': layer.weights,
                'bias': layer.bias,
                'activation': layer.activation
            })
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Modelo guardado en {filepath}")
    
    def load(self, filepath):
        """Cargar el modelo"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.architecture = model_data['architecture']
        self.training_history = model_data['training_history']
        self.layers = []
        
        for layer_data in model_data['layers']:
            layer = NeuralLayer(1, 1)  # Tamaño temporal
            layer.weights = layer_data['weights']
            layer.bias = layer_data['bias']
            layer.activation = layer_data['activation']
            self.layers.append(layer)
        
        print(f"Modelo cargado desde {filepath}")


class NexusAI:
    """
    Sistema de IA principal de NEXUS
    Integra búsqueda inteligente, aprendizaje y ranking
    """
    
    def __init__(self, model_path='./models/nexus_brain.pkl'):
        self.model_path = model_path
        
        # Arquitectura: 50 neuronas en capas ocultas
        # Input: 512 (embeddings de búsqueda)
        # Hidden: 256 -> 128 -> 64 -> 50 -> 32
        # Output: 10 (categorías de relevancia)
        self.architecture = [512, 256, 128, 64, 50, 32, 10]
        
        self.model = DeepNeuralNetwork(self.architecture)
        
        # Intentar cargar modelo existente
        if os.path.exists(model_path):
            self.model.load(model_path)
        
        # Memoria de búsquedas
        self.search_memory = []
        self.user_preferences = {}
        
    def vectorize_query(self, query: str) -> np.ndarray:
        """Convertir query a vector de 512 dimensiones"""
        # Hash del query
        query_hash = hashlib.sha256(query.encode()).digest()
        
        # Crear vector base
        vector = np.zeros(512)
        
        # Llenar con características
        words = query.lower().split()
        for i, word in enumerate(words[:50]):
            # Convertir palabra a número
            word_value = sum(ord(c) for c in word)
            idx = (word_value * (i + 1)) % 512
            vector[idx] = 1.0
        
        # Agregar características de hash
        for i, byte in enumerate(query_hash[:462]):
            vector[i + 50] = byte / 255.0
        
        return vector.reshape(1, -1)
    
    def predict_relevance(self, query: str, results: List[Dict]) -> List[float]:
        """Predecir relevancia de resultados para un query"""
        query_vector = self.vectorize_query(query)
        
        relevance_scores = []
        for result in results:
            # Combinar query con features del resultado
            result_text = f"{result.get('title', '')} {result.get('description', '')}"
            result_vector = self.vectorize_query(result_text)
            
            # Concatenar vectores
            combined = np.concatenate([query_vector, result_vector], axis=1)[:, :512]
            
            # Predecir
            prediction = self.model.predict(combined)
            relevance = float(np.max(prediction))
            relevance_scores.append(relevance)
        
        return relevance_scores
    
    def learn_from_click(self, query: str, clicked_result: Dict, position: int):
        """Aprender de los clicks del usuario"""
        # Crear datos de entrenamiento
        query_vector = self.vectorize_query(query)
        
        # Etiqueta: alta relevancia si se clickeó en las primeras posiciones
        relevance = 1.0 / (position + 1)
        label = np.zeros(10)
        label[int(relevance * 9)] = 1.0
        
        # Entrenar con este ejemplo
        self.model.train(query_vector, label.reshape(1, -1), epochs=1, verbose=False)
        
        # Guardar en memoria
        self.search_memory.append({
            'query': query,
            'result': clicked_result,
            'position': position,
            'timestamp': datetime.now().isoformat()
        })
        
        # Guardar modelo cada 100 interacciones
        if len(self.search_memory) % 100 == 0:
            self.save_model()
    
    def save_model(self):
        """Guardar el modelo y la memoria"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        
        # Guardar memoria
        memory_path = self.model_path.replace('.pkl', '_memory.json')
        with open(memory_path, 'w') as f:
            json.dump(self.search_memory[-1000:], f)  # Solo últimas 1000
    
    def get_personalized_results(self, query: str, results: List[Dict], user_id: str = None) -> List[Dict]:
        """Reordenar resultados según aprendizaje y preferencias del usuario"""
        # Predecir relevancia
        relevance_scores = self.predict_relevance(query, results)
        
        # Si hay preferencias de usuario, ajustar scores
        if user_id and user_id in self.user_preferences:
            prefs = self.user_preferences[user_id]
            for i, result in enumerate(results):
                # Ajustar por categorías preferidas
                if result.get('category') in prefs.get('favorite_categories', []):
                    relevance_scores[i] *= 1.5
        
        # Combinar con score original
        for i, result in enumerate(results):
            original_score = result.get('score', 0.5)
            ai_score = relevance_scores[i]
            # 70% IA, 30% score original
            result['final_score'] = (ai_score * 0.7) + (original_score * 0.3)
        
        # Ordenar por score final
        sorted_results = sorted(results, key=lambda x: x['final_score'], reverse=True)
        
        return sorted_results


# Función para crear y entrenar el modelo inicial
def initialize_nexus_ai():
    """Inicializar y pre-entrenar el sistema de IA"""
    print("🧠 Inicializando NEXUS AI...")
    
    nexus = NexusAI()
    
    # Datos de entrenamiento sintéticos
    print("📚 Generando datos de entrenamiento...")
    training_queries = [
        "python tutorial", "javascript frameworks", "machine learning",
        "web development", "data science", "react hooks",
        "nodejs express", "mongodb tutorial", "docker containers",
        "kubernetes deployment", "tensorflow examples", "neural networks"
    ]
    
    X_train = []
    y_train = []
    
    for query in training_queries:
        vector = nexus.vectorize_query(query)
        X_train.append(vector[0])
        
        # Etiqueta sintética
        label = np.zeros(10)
        label[np.random.randint(0, 10)] = 1.0
        y_train.append(label)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Entrenar
    print("🎯 Entrenando red neuronal...")
    nexus.model.train(X_train, y_train, epochs=50, batch_size=4)
    
    # Guardar
    nexus.save_model()
    print("✅ NEXUS AI inicializado y guardado!")
    
    return nexus


if __name__ == "__main__":
    # Inicializar sistema
    nexus = initialize_nexus_ai()
    
    # Ejemplo de uso
    print("\n🔍 Ejemplo de predicción:")
    query = "python machine learning tutorial"
    results = [
        {'title': 'Python ML Guide', 'description': 'Complete guide to ML with Python', 'score': 0.8},
        {'title': 'JavaScript Basics', 'description': 'Learn JavaScript fundamentals', 'score': 0.3},
        {'title': 'Advanced Python', 'description': 'Advanced Python techniques', 'score': 0.7}
    ]
    
    ranked_results = nexus.get_personalized_results(query, results)
    
    print(f"\nQuery: {query}")
    print("\nResultados rankeados por IA:")
    for i, result in enumerate(ranked_results, 1):
        print(f"{i}. {result['title']} - Score: {result['final_score']:.3f}")
