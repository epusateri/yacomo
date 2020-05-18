clean:
	rm -rf build/ dist/

dev_env:
	conda create --prefix=dev_env python=3.7.7

dev_install_no_deps: dev_env
	pip install --upgrade --force-reinstall --no-deps -e .

dev_install: dev_env
	pip install --upgrade --force-reinstall -e .
