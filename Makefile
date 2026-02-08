# Define variables
VENV_DIR = venv
PYTHON = python3
PIP = $(VENV_DIR)/bin/pip

# Default target
all: install

# Create virtual environment
$(VENV_DIR):
	$(PYTHON) -m venv $(VENV_DIR)

# Install requirements
install: $(VENV_DIR)
	$(PIP) install -r requirements.txt

# Remove the virtual environment
clean:
	rm -rf $(VENV_DIR)

.PHONY: all install clean
