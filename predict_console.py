import joblib
import pandas as pd
import os

def get_number_input(prompt):
    """Функция для валидации ввода: только числа."""
    while True:
        user_input = input(prompt)
        try:
            return float(user_input)
        except ValueError:
            print("Ошибка: введите число")

def main():
    # Проверка наличия файлов
    if not os.path.exists('kmeans_model.pkl') or not os.path.exists('scaler.pkl'):
        print("Ошибка: Файлы модели не найдены. Сначала запустите train_model.py")
        return

    # Загрузка
    model = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')

    print("\n=== Система предсказания кластера транзакции ===")
    
    # Ввод данных пользователем
    amount = get_number_input("Введите сумму транзакции: ")
    longitude = get_number_input("Введите долготу (Long): ")
    latitude = get_number_input("Введите широту (Lat): ")

   
    new_data = pd.DataFrame(
        [[amount, longitude, latitude]],
        columns=["amount", "long", "lat"]
    )
  
    new_data_scaled = scaler.transform(new_data)

    cluster_id = model.predict(new_data_scaled)[0]

    print(f"\n[Результат] Данная транзакция относится к кластеру: {cluster_id}")

if __name__ == "__main__":
    main()