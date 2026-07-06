.PHONY: help install run-api run-ui run clean

help:
	@echo "install    Install dependencies into system Python"
	@echo "run-api    Start FastAPI on :8000"
	@echo "run-ui     Start Streamlit on :8501"
	@echo "run        Start both (api in background)"
	@echo "clean      Remove caches"

install:
	pip3 install --break-system-packages -r requirements.txt

run-api:
	PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	PYTHONPATH=. streamlit run ui/app.py --server.port 8501

run:
	PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
	PYTHONPATH=. streamlit run ui/app.py --server.port 8501

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
