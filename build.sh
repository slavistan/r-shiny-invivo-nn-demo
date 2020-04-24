#!/usr/bin/env sh

case $1 in
  watch) R -e "options(shiny.autoreload=T);shiny::runApp('./visualize-mnist.r')";;
esac

