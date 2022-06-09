#!/bin/bash

TOKEN=$1

if [ -f ~/.config/pip/pip.conf ]; then
	echo "pip.conf already exists, you've to add python indices manually."
else
	echo "creating pip.conf..."
	mkdir -p ~/.config/pip
	cat <<- EOF > ~/.config/pip/pip.conf
	[global]
	extra-index-url = https://__token__:${TOKEN}@git.eodc.eu/api/v4/projects/581/packages/pypi/simple
	EOF
fi