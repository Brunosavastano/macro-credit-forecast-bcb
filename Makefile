PYTHON ?= python

.PHONY: refresh forecast backtest audit app test all

refresh:
	$(PYTHON) -m macro_credit_forecast_bcb.pipeline.refresh

forecast:
	$(PYTHON) -m macro_credit_forecast_bcb.pipeline.forecast

backtest:
	$(PYTHON) -m macro_credit_forecast_bcb.pipeline.backtest

audit:
	$(PYTHON) -m macro_credit_forecast_bcb.pipeline.audit

app:
	streamlit run app/streamlit_app.py

test:
	pytest

all: refresh forecast backtest audit test
