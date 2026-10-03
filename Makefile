HUGO ?= hugo
PYTHON ?= python3

BASE_URL ?= https://datahatchery.com/blog/
BUILD_DIR ?= build
PREVIEW_DIR ?= build-preview
PORT ?= 1315

.PHONY: build preview

build:
	$(HUGO) --gc --minify --cleanDestinationDir --baseURL '$(BASE_URL)' --destination '$(BUILD_DIR)/blog'
	cp CNAME '$(BUILD_DIR)/CNAME'
	cp deploy/root-redirect.html '$(BUILD_DIR)/index.html'

preview:
	$(HUGO) --gc --minify --cleanDestinationDir --buildDrafts --baseURL 'http://localhost:$(PORT)/blog/' --destination '$(PREVIEW_DIR)/blog'
	cp CNAME '$(PREVIEW_DIR)/CNAME'
	cp deploy/root-redirect.html '$(PREVIEW_DIR)/index.html'
	$(PYTHON) -m http.server $(PORT) --directory '$(PREVIEW_DIR)'
