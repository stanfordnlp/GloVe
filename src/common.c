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

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"

#ifdef _MSC_VER
#define STRERROR(ERRNO, BUF, BUFSIZE) strerror_s((BUF), (BUFSIZE), (ERRNO))
#else
#define STRERROR(ERRNO, BUF, BUFSIZE) strerror_r((ERRNO), (BUF), (BUFSIZE))
#endif

/* Efficient string comparison */
int scmp( char *s1, char *s2 ) {
    while (*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return (*s1 - *s2);
}

/* Move-to-front hashing and hash function from Hugh Williams, http://www.seg.rmit.edu.au/code/zwh-ipl/ */

/* Simple bitwise hash function */
unsigned int bitwisehash(char *word, int tsize, unsigned int seed) {
    char c;
    unsigned int h;
    h = seed;
    for ( ; (c = *word) != '\0'; word++) h ^= ((h << 5) + c + (h >> 2));
    return (unsigned int)((h & 0x7fffffff) % tsize);
}

/* Create hash table, initialise pointers to NULL */
HASHREC ** inithashtable() {
    int i;
    HASHREC **ht;
    ht = (HASHREC **) malloc( sizeof(HASHREC *) * TSIZE );
    for (i = 0; i < TSIZE; i++) ht[i] = (HASHREC *) NULL;
    return ht;
}

/* Read word from input stream. Return 1 when encounter '\n' or EOF (but separate from word), 0 otherwise.
   Words can be separated by space(s), tab(s), or newline(s). Carriage return characters are just ignored.
   (Okay for Windows, but not for Mac OS 9-. Ignored even if by themselves or in words.)
   A newline is taken as indicating a new document (contexts won't cross newline).
   Argument word array is assumed to be of size MAX_STRING_LENGTH.
   words will be truncated if too long. They are truncated with some care so that they
   cannot truncate in the middle of a utf-8 character, but
   still little to no harm will be done for other encodings like iso-8859-1.
   (This function appears identically copied in vocab_count.c and cooccur.c.)
 */
int get_word(char *word, FILE *fin) {
    int i = 0, ch;
    for ( ; ; ) {
        ch = fgetc(fin);
        if (ch == '\r') continue;
        if (i == 0 && ((ch == '\n') || (ch == EOF))) {
            word[i] = 0;
            return 1;
        }
        if (i == 0 && ((ch == ' ') || (ch == '\t'))) continue; // skip leading space
        if ((ch == EOF) || (ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (ch == '\n') ungetc(ch, fin); // return the newline next time as document ender
            break;
        }
        if (i < MAX_STRING_LENGTH - 1)
          word[i++] = ch; // don't allow words to exceed MAX_STRING_LENGTH
    }
    word[i] = 0; //null terminate
    // avoid truncation destroying a multibyte UTF-8 char except if only thing on line (so the i > x tests won't overwrite word[0])
    // see https://en.wikipedia.org/wiki/UTF-8#Description
    if (i == MAX_STRING_LENGTH - 1 && (word[i-1] & 0x80) == 0x80) {
        if ((word[i-1] & 0xC0) == 0xC0) {
            word[i-1] = '\0';
        } else if (i > 2 && (word[i-2] & 0xE0) == 0xE0) {
            word[i-2] = '\0';
        } else if (i > 3 && (word[i-3] & 0xF8) == 0xF0) {
            word[i-3] = '\0';
        }
    }
    return 0;
}

int find_arg(char *str, int argc, char **argv) {
    int i;
    for (i = 1; i < argc; i++) {
        if (!scmp(str, argv[i])) {
            if (i == argc - 1) {
                printf("No argument given for %s\n", str);
                exit(1);
            }
            return i;
        }
    }
    return -1;
}

void free_table(HASHREC **ht) {
    int i;
    HASHREC* current;
    HASHREC* tmp;
    for (i = 0; i < TSIZE; i++) {
        current = ht[i];
        while (current != NULL) {
            tmp = current;
            current = current->next;
            free(tmp->word);
            free(tmp);
        }
    }
    free(ht);
}

void free_fid(FILE **fid, const int num) {
    int i;
    for(i = 0; i < num; i++) {
        if(fid[i] != NULL)
            fclose(fid[i]);
    }
    free(fid);
}


int log_file_loading_error(char *file_description, char *file_name) {
    fprintf(stderr, "Unable to open %s %s.\n", file_description, file_name);
    fprintf(stderr, "Errno: %d\n", errno);
    char error[MAX_STRING_LENGTH];
    STRERROR(errno, error, MAX_STRING_LENGTH);
    fprintf(stderr, "Error description: %s\n", error);
    return errno;
}
