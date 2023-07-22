#
set -e

#
black src/veritas
black src/test

#
mypy

#
rm -rf debug/*
pytest
