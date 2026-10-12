.PHONY: test benchmark serve
test:
	pytest -q
benchmark:
	python -m refundops_rl.benchmark --episodes 60 --policy scripted --split eval --output examples/rollouts.jsonl
serve:
	uvicorn refundops_rl.server:app --reload --host 127.0.0.1 --port 8000
