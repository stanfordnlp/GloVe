//  Tool to shuffle entries of word-word cooccurrence files
//
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
//    GlobalVectors@googlegroups.com
//    http://nlp.stanford.edu/projects/glove/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "common.h"


static const long LRAND_MAX = ((long) RAND_MAX + 2) * (long)RAND_MAX;

int verbose = 2; // 0, 1, or 2
int seed = 0;
long long array_size = 2000000; // size of chunks to shuffle individually
char *file_head; // temporary file string
real memory_limit = 2.0; // soft limit, in gigabytes

/* Generate uniformly distributed random long ints */
static long rand_long(long n) {
    long limit = LRAND_MAX - LRAND_MAX % n;
    long rnd;
    do {
        rnd = ((long)RAND_MAX + 1) * (long)rand() + (long)rand();
    } while (rnd >= limit);
    return rnd % n;
}

/* Write contents of array to binary file */
int write_chunk(CREC *array, long size, FILE *fout) {
    long i = 0;
    for (i = 0; i < size; i++) fwrite(&array[i], sizeof(CREC), 1, fout);
    return 0;
}

/* Fisher-Yates shuffle */
void shuffle(CREC *array, long n) {
    long i, j;
    CREC tmp;
    for (i = n - 1; i > 0; i--) {
        j = rand_long(i + 1);
        tmp = array[j];
        array[j] = array[i];
        array[i] = tmp;
    }
}

/* Merge shuffled temporary files; doesn't necessarily produce a perfect shuffle, but good enough */
int shuffle_merge(int num) {
    long i, j, k, l = 0;
    int fidcounter = 0;
    CREC *array;
    char filename[MAX_STRING_LENGTH];
    FILE **fid, *fout = stdout;
    
    array = malloc(sizeof(CREC) * array_size);
    fid = calloc(num, sizeof(FILE));
    for (fidcounter = 0; fidcounter < num; fidcounter++) { //num = number of temporary files to merge
        sprintf(filename,"%s_%04d.bin",file_head, fidcounter);
        fid[fidcounter] = fopen(filename, "rb");
        if (fid[fidcounter] == NULL) {
            log_file_loading_error("temp file", filename);
            free(array);
            free_fid(fid, num);
            return 1;
        }
    }
    if (verbose > 0) fprintf(stderr, "Merging temp files: processed %ld lines.", l);
    
    while (1) { //Loop until EOF in all files
        i = 0;
        //Read at most array_size values into array, roughly array_size/num from each temp file
        for (j = 0; j < num; j++) {
            if (feof(fid[j])) continue;
            for (k = 0; k < array_size / num; k++){
                fread(&array[i], sizeof(CREC), 1, fid[j]);
                if (feof(fid[j])) break;
                i++;
            }
        }
        if (i == 0) break;
        l += i;
        shuffle(array, i); // Shuffles lines between temp files
        write_chunk(array,i,fout);
        if (verbose > 0) fprintf(stderr, "\033[31G%ld lines.", l);
    }
    fprintf(stderr, "\033[0GMerging temp files: processed %ld lines.", l);
    for (fidcounter = 0; fidcounter < num; fidcounter++) {
        fclose(fid[fidcounter]);
        sprintf(filename,"%s_%04d.bin",file_head, fidcounter);
        remove(filename);
    }
    fprintf(stderr, "\n\n");
    free(array);
    free(fid);
    return 0;
}

/* Shuffle large input stream by splitting into chunks */
int shuffle_by_chunks() {
    if (seed == 0) {
        seed = time(0);
    }
    fprintf(stderr, "Using random seed %d\n", seed);
    srand(seed);
    long i = 0, l = 0;
    int fidcounter = 0;
    char filename[MAX_STRING_LENGTH];
    CREC *array;
    FILE *fin = stdin, *fid;
    array = malloc(sizeof(CREC) * array_size);
    
    fprintf(stderr,"SHUFFLING COOCCURRENCES\n");
    if (verbose > 0) fprintf(stderr,"array size: %lld\n", array_size);
    sprintf(filename,"%s_%04d.bin",file_head, fidcounter);
    fid = fopen(filename,"w");
    if (fid == NULL) {
        log_file_loading_error("file", filename);
        free(array);
        return 1;
    }
    if (verbose > 1) fprintf(stderr, "Shuffling by chunks: processed 0 lines.");
    
    while (1) { //Continue until EOF
        if (i >= array_size) {// If array is full, shuffle it and save to temporary file
            shuffle(array, i);
            l += i;
            if (verbose > 1) fprintf(stderr, "\033[22Gprocessed %ld lines.", l);
            write_chunk(array,i,fid);
            fclose(fid);
            fidcounter++;
            sprintf(filename,"%s_%04d.bin",file_head, fidcounter);
            fid = fopen(filename,"w");
            if (fid == NULL) {
                log_file_loading_error("file", filename);
                free(array);
                return 1;
            }
            i = 0;
        }
        fread(&array[i], sizeof(CREC), 1, fin);
        if (feof(fin)) break;
        i++;
    }
    shuffle(array, i); //Last chunk may be smaller than array_size
    write_chunk(array,i,fid);
    l += i;
    if (verbose > 1) fprintf(stderr, "\033[22Gprocessed %ld lines.\n", l);
    if (verbose > 1) fprintf(stderr, "Wrote %d temporary file(s).\n", fidcounter + 1);
    fclose(fid);
    free(array);
    return shuffle_merge(fidcounter + 1); // Merge and shuffle together temporary files
}

int main(int argc, char **argv) {
    int i;
    
    if (argc == 2 &&
        (!scmp(argv[1], "-h") || !scmp(argv[1], "-help") || !scmp(argv[1], "--help"))) {
        printf("Tool to shuffle entries of word-word cooccurrence files\n");
        printf("Author: Jeffrey Pennington (jpennin@stanford.edu)\n\n");
        printf("Usage options:\n");
        printf("\t-verbose <int>\n");
        printf("\t\tSet verbosity: 0, 1, or 2 (default)\n");
        printf("\t-memory <float>\n");
        printf("\t\tSoft limit for memory consumption, in GB; default 4.0\n");
        printf("\t-array-size <int>\n");
        printf("\t\tLimit to length <int> the buffer which stores chunks of data to shuffle before writing to disk. \n\t\tThis value overrides that which is automatically produced by '-memory'.\n");
        printf("\t-temp-file <file>\n");
        printf("\t\tFilename, excluding extension, for temporary files; default temp_shuffle\n");
        printf("\t-seed <int>\n");
        printf("\t\tRandom seed to use.  If not set, will be randomized using current time.");
        printf("\nExample usage: (assuming 'cooccurrence.bin' has been produced by 'coccur')\n");
        printf("./shuffle -verbose 2 -memory 8.0 < cooccurrence.bin > cooccurrence.shuf.bin\n");
        return 0;
    }

    file_head = malloc(sizeof(char) * MAX_STRING_LENGTH);
    if ((i = find_arg((char *)"-verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-temp-file", argc, argv)) > 0) strcpy(file_head, argv[i + 1]);
    else strcpy(file_head, (char *)"temp_shuffle");
    if ((i = find_arg((char *)"-memory", argc, argv)) > 0) memory_limit = atof(argv[i + 1]);
    array_size = (long long) (0.95 * (real)memory_limit * 1073741824/(sizeof(CREC)));
    if ((i = find_arg((char *)"-array-size", argc, argv)) > 0) array_size = atoll(argv[i + 1]);
    if ((i = find_arg((char *)"-seed", argc, argv)) > 0) seed = atoi(argv[i + 1]);
    const int returned_value = shuffle_by_chunks();
    free(file_head);
    return returned_value;
}

