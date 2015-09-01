CC = gcc
#For older gcc, use -O3 or -O2 instead of -Ofast
CFLAGS = -lm -pthread -Ofast -march=native -funroll-loops -Wno-unused-result
BUILDDIR := build

all: dir glove shuffle cooccur vocab_count

dir :
	mkdir -p $(BUILDDIR)
glove : glove.c
	$(CC) glove.c -o $(BUILDDIR)/glove $(CFLAGS)
shuffle : shuffle.c
	$(CC) shuffle.c -o $(BUILDDIR)/shuffle $(CFLAGS)
cooccur : cooccur.c
	$(CC) cooccur.c -o $(BUILDDIR)/cooccur $(CFLAGS)
vocab_count : vocab_count.c
	$(CC) vocab_count.c -o $(BUILDDIR)/vocab_count $(CFLAGS)

clean:
	rm -rf glove shuffle cooccur vocab_count build
