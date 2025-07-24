# fix_sentiment_model.py
def fix_mathematical_sentiment_model():
    # Читаем файл
    with open('mathematical_sentiment_model.py', 'r') as f:
        content = f.read()
    
    # Заменяем конструктор
    old_init = "def __init__(self):"
    new_init = "def __init__(self, custom_weights: dict = None, custom_reliabilities: dict = None):"
    content = content.replace(old_init, new_init)
    
    # Заменяем вызов _initialize_sources
    old_call = "self.sources = self._initialize_sources()"
    new_call = "self.sources = self._initialize_sources(custom_weights, custom_reliabilities)"
    content = content.replace(old_call, new_call)
    
    # Обновляем _initialize_sources метод
    old_method = 'def _initialize_sources(self) -> dict:'
    new_method = 'def _initialize_sources(self, custom_weights: dict = None, custom_reliabilities: dict = None) -> dict:'
    content = content.replace(old_method, new_method)
    
    # Сохраняем
    with open('mathematical_sentiment_model.py', 'w') as f:
        f.write(content)
    
    print("✅ mathematical_sentiment_model.py обновлен!")

if __name__ == "__main__":
    fix_mathematical_sentiment_model()