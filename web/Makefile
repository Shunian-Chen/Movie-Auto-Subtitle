.PHONY: run run-container gcloud-deploy

run:
	@python web_app.py --port=8080 --address=127.0.0.1

run-container:
	@docker build . -t app.py
	@docker run -p 8080:8080 app.py

gcloud-deploy:
	@gcloud config set project movie-auto_subtitle
	@gcloud app deploy app.yaml --stop-previous-version

