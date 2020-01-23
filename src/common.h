#ifndef COMMON_H
#define COMMON_H

//  Common code for cooccur.c, vocab_count.c,
//  glove.c and shuffle.c
//
//  GloVe: Global Vectors for Word Representation
//  Copyright (c) 2014 The Board of Trustees of
//  The Leland Stanford Junior University. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//
//  For more information, bug reports, fixes, contact:
//    Jeffrey Pennington (jpennin@stanford.edu)
//    Christopher Manning (manning@cs.stanford.edu)
//    https://github.com/stanfordnlp/GloVe/
//    GlobalVectors@googlegroups.com
//    http://nlp.stanford.edu/projects/glove/

#include <stdio.h>

#define MAX_STRING_LENGTH 1000
#define TSIZE 1048576
#define SEED 1159241
#define HASHFN bitwisehash

typedef double real;
typedef struct cooccur_rec {
    int word1;
    int word2;
    real val;
} CREC;
typedef struct hashrec {
    char *word;
    long long num; //count or id
    struct hashrec *next;
} HASHREC;


int scmp( char *s1, char *s2 );
unsigned int bitwisehash(char *word, int tsize, unsigned int seed);
HASHREC **inithashtable();
int get_word(char *word, FILE *fin);
void free_table(HASHREC **ht);
int find_arg(char *str, int argc, char **argv);
void free_fid(FILE **fid, const int num);

// logs errors when loading files.  call after a failed load
int log_file_loading_error(char *file_description, char *file_name);

#endif /* COMMON_H */

