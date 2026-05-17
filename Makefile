PYTHON ?= python

.PHONY: refresh forecast backtest app test all

refresh:
	$(PYTHON) -m macro_credit_forecast_bcb.pipeline.refresh

forecast:
	$(PYTHON) -m macro_credit_forecast_bcb.pipeline.forecast

backtest:
	$(PYTHON) -m macro_credit_forecast_bcb.pipeline.backtest

app:
	streamlit run app/streamlit_app.py

test:
	pytest

all: refresh forecast backtest test

