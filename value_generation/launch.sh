#!/bin/bash
parallel --no-notice -j12 --ungroup python value_generation.py ../data/*.pgn ::: {1..12}
