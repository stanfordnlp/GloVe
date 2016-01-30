# To enable nmake and cl locate and run "vcvarsall.bat amd64"
# typically in "C:\Program Files (x86)\Microsoft Visual Studio 99.9\VC"
# see: https://msdn.microsoft.com/en-us/library/x4d2c09s.aspx

CC=CL
# optimization flags may not be the best
# see /arch flag here: https://msdn.microsoft.com/en-us/library/jj620901.aspx
CFLAGS=/TC /MT /W3 /D_CRT_SECURE_NO_WARNINGS /Ox /Fp:fast /arch:AVX2
BUILDDIR=build
SRCDIR=src

all: dir vocab_count cooccur shuffle glove

dir :
	IF exist $(BUILDDIR) ( echo $(BUILDDIR) exists ) ELSE ( MD $(BUILDDIR))
glove : $(SRCDIR)/glove.c
	$(CC) $(SRCDIR)/glove.c /Fo$(BUILDDIR)/glove.obj $(CFLAGS) /link /OUT:$(BUILDDIR)/glove.exe
shuffle : $(SRCDIR)/shuffle.c
	$(CC) $(SRCDIR)/shuffle.c /Fo$(BUILDDIR)/shuffle.obj $(CFLAGS) /link /OUT:$(BUILDDIR)/shuffle.exe
cooccur : $(SRCDIR)/cooccur.c
	$(CC) $(SRCDIR)/cooccur.c /Fo$(BUILDDIR)/cooccur.obj $(CFLAGS) /link /OUT:$(BUILDDIR)/cooccur.exe
vocab_count : $(SRCDIR)/vocab_count.c
	$(CC) $(SRCDIR)/vocab_count.c /Fo$(BUILDDIR)/vocab_count.obj $(CFLAGS) /link /OUT:$(BUILDDIR)/vocab_count.exe

clean:
	RD /Q /S $(BUILDDIR)
