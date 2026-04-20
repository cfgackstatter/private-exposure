.PHONY: dev backend frontend db-start db-stop

dev: backend frontend

backend:
	cd src && uvicorn private_exposure.main:app --reload

frontend:
	cd frontend && npm run dev

db-start:
	sudo service postgresql start

db-stop:
	sudo service postgresql stop

db-status:
	sudo service postgresql status

install:
	pip install -e "src/[dev]"
	cd frontend && npm install