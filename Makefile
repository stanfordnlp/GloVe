CC = gcc
#For older gcc, use -O3 or -O2 instead of -Ofast
CFLAGS = -lm -pthread -Ofast -march=native -funroll-loops -Wno-unused-result

all: glove shuffle cooccur vocab_count

glove : glove.c
	$(CC) glove.c -o glove $(CFLAGS)
shuffle : shuffle.c
	$(CC) shuffle.c -o shuffle $(CFLAGS)
cooccur : cooccur.c
	$(CC) cooccur.c -o cooccur $(CFLAGS)
vocab_count : vocab_count.c
	$(CC) vocab_count.c -o vocab_count $(CFLAGS)

clean:
	rm -rf glove shuffle cooccur vocab_count
