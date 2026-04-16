import numpy as np
from model import train_and_predict, get_accuracy

def test_predictions_not_none():
    """Sprawdzamy, czy otrzymujemy jakąkolwiek predykcję."""
    preds, _ = train_and_predict()
    assert preds is not None, "Predictions should not be None."

def test_predictions_length():
    """Sprawdzamy, czy długość listy predykcji jest większa od 0 i czy odpowiada przewidywanej liczbie próbek testowych."""
    preds, y_test = train_and_predict()
    assert len(preds) > 0, "Długość listy predykcji musi być większa od 0"
    assert len(preds) == len(y_test), "Liczba predykcji nie zgadza się z liczbą próbek testowych"

def test_predictions_value_range():
    """Sprawdzamy, czy wartości w predykcjach mieszczą się w spodziewanym zakresie: Dla zbioru Iris mamy 3 klasy (0, 1, 2)."""
    preds, _ = train_and_predict()
    dozwolone_klasy = {0, 1, 2}
    # Upewniamy się, że zbiór unikalnych wartości predykcji zawiera się w dozwolonych klasach
    assert set(preds).issubset(dozwolone_klasy), f"Predykcje poza zakresem! Otrzymano: {set(preds)}"

def test_model_accuracy():
    """Sprawdzamy, czy model osiąga co najmniej 70% dokładności."""
    accuracy = get_accuracy()
    assert accuracy >= 0.70, f"Dokładność modelu jest zbyt niska: {accuracy:.2f}"