# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#!/bin/sh

../fastText/fasttext print-word-vectors ../Embeddings/wiki.en.bin < ./data/$1-en.txt > ./data/$1-en-emb.vec
../fastText/fasttext print-word-vectors ../Embeddings/wiki.es.bin < ./data/$1-es.txt > ./data/$1-es-emb.vec

