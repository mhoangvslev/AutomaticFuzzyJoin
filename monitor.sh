#!/bin/bash

mkdir -p psrecord/
for proc in $(pgrep python)
do
	echo "Recording proc $proc..."
	psrecord $proc --interval 1 --plot "psrecord/$proc.png" &
done
