#!/bin/bash
find ./coco2/labels -name '*.txt' -exec awk -i inplace '!seen[$0]++' {} \;