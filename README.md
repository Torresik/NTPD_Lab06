## Wykorzystane technologie i narzędzia:
* **Model ML i dane:** Biblioteka `scikit-learn` (prosty model klasyfikacji wytrenowany na zbiorze Iris).
* **API:** `FastAPI` oraz serwer `Uvicorn` do udostępniania predykcji i metryk modelu.
* **Testowanie (CI):** Biblioteka `pytest` do automatycznych testów jednostkowych.
* **Konteneryzacja:** `Docker` (plik Dockerfile budujący lekkie środowisko oparte na Python 3.9-slim).
* **Automatyzacja (CI/CD):** `GitHub Actions` – zdefiniowany potok automatycznie uruchamia środowisko, instaluje zależności, wykonuje testy `pytest`, a następnie buduje obraz kontenera przy każdym pushu na gałąź główną.
* **Publikacja obrazu:** Zbudowany obraz jest automatycznie wysyłany do `GitHub Container Registry`.