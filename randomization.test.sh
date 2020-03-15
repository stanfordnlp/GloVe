# Tests for ensuring randomization is being controlled

make

if [ ! -e text8 ]; then
  if hash wget 2>/dev/null; then
    wget http://mattmahoney.net/dc/text8.zip
  else
    curl -O http://mattmahoney.net/dc/text8.zip
  fi
  unzip text8.zip
  rm text8.zip
fi

# Global constants
CORPUS=text8
VERBOSE=2
BUILDDIR=build
MEMORY=4.0
VOCAB_MIN_COUNT=20

# Re-used files
VOCAB_FILE=$(mktemp vocab.test.txt.XXXXXX)
COOCCURRENCE_FILE=$(mktemp cooccurrence.test.bin.XXXXXX)
COOCCURRENCE_SHUF_FILE=$(mktemp cooccurrence_shuf.test.bin.XXXXXX)

# Make vocab
$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE

# Make Coocurrences
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size 5 < $CORPUS > $COOCCURRENCE_FILE

# Shuffle Coocurrences
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE -seed 1 < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE

# Keep track of failure
num_failed=0

check_exit() {
  eval $2
  failed=$(( $1 != $? ))
  num_failed=$(( $num_failed + $failed ))
  if [[ $failed -eq 0 ]]; then
    echo PASSED
  else
    echo FAILED
  fi
}

# Test control of random seed in shuffle
printf "\n\n--- TEST SET: Control of random seed in shuffle\n"
TEST_FILE=$(mktemp cooc_shuf.test.bin.XXXXXX)

printf "\n- TEST: Using the same seed should get the same shuffle\n"
$BUILDDIR/shuffle -memory $MEMORY -verbose 0 -seed 1 < $COOCCURRENCE_FILE > $TEST_FILE
check_exit 0 "cmp --quiet $COOCCURRENCE_SHUF_FILE $TEST_FILE"

printf "\n- TEST: Changing the seed should change the shuffle\n"
$BUILDDIR/shuffle -memory $MEMORY -verbose 0 -seed 2 < $COOCCURRENCE_FILE > $TEST_FILE
check_exit 1 "cmp --quiet $COOCCURRENCE_SHUF_FILE $TEST_FILE"

rm $TEST_FILE  # Clean up
# ---

# Control randomization in GloVe
printf "\n\n--- TEST SET: Control of random seed in glove\n"
# Note "-threads" must equal 1 for these to pass, since order in which results come back from individual threads is uncontrolled
BASE_PREFIX=$(mktemp base_vectors.XXXXXX)
TEST_PREFIX=$(mktemp test_vectors.XXXXXX)

printf "\n- TEST: Reusing seed should give the same vectors\n"
$BUILDDIR/glove -save-file $BASE_PREFIX -threads 1 -input-file $COOCCURRENCE_SHUF_FILE -iter 3 -vector-size 10 -binary 1 -vocab-file $VOCAB_FILE -verbose 0 -seed 1
$BUILDDIR/glove -save-file $TEST_PREFIX -threads 1 -input-file $COOCCURRENCE_SHUF_FILE -iter 3 -vector-size 10 -binary 1 -vocab-file $VOCAB_FILE -verbose 0 -seed 1
check_exit 0 "cmp --quiet $BASE_PREFIX.bin $TEST_PREFIX.bin"

printf "\n- TEST: Changing seed should change the learned vectors\n"
$BUILDDIR/glove -save-file $TEST_PREFIX -threads 1 -input-file $COOCCURRENCE_SHUF_FILE -iter 3 -vector-size 10 -binary 1 -vocab-file $VOCAB_FILE -verbose 0 -seed 2
check_exit 1 "cmp --quiet $BASE_PREFIX.bin $TEST_PREFIX.bin"

printf "\n- TEST: Should be able to save/load initial parameters\n"
$BUILDDIR/glove -save-file $BASE_PREFIX -threads 1 -input-file $COOCCURRENCE_SHUF_FILE -iter 3 -vector-size 10 -binary 1 -vocab-file $VOCAB_FILE -verbose 0 -save-init-param 1
$BUILDDIR/glove -save-file $TEST_PREFIX -threads 1 -input-file $COOCCURRENCE_SHUF_FILE -iter 3 -vector-size 10 -binary 1 -vocab-file $VOCAB_FILE -verbose 0 -save-init-param 1 -load-init-param 1 -init-param-file "$BASE_PREFIX.000.bin"
check_exit 0 "cmp --quiet $BASE_PREFIX.000.bin $TEST_PREFIX.000.bin && cmp --quiet $BASE_PREFIX.bin $TEST_PREFIX.bin"

rm "$BASE_PREFIX.000.bin" "$TEST_PREFIX.000.bin" "$BASE_PREFIX.bin" "$TEST_PREFIX.bin" # Clean up
rm $BASE_PREFIX $TEST_PREFIX

# ----

printf "\n- TEST: Should be able to save/load initial parameters and gradsq\n"
# note: the seed will be randomly assigned and should not matter
$BUILDDIR/glove -save-file $BASE_PREFIX -gradsq-file $BASE_PREFIX.gradsq -threads 1 -input-file $COOCCURRENCE_SHUF_FILE -iter 6 -vector-size 10 -binary 1 -vocab-file $VOCAB_FILE -verbose 0 -checkpoint-every 2

$BUILDDIR/glove -save-file $TEST_PREFIX -gradsq-file $TEST_PREFIX.gradsq -threads 1 -input-file $COOCCURRENCE_SHUF_FILE -iter 4 -vector-size 10 -binary 1 -vocab-file $VOCAB_FILE -verbose 0 -checkpoint-every 2 -load-init-param 1 -init-param-file "$BASE_PREFIX.002.bin" -load-init-gradsq 1 -init-gradsq-file "$BASE_PREFIX.gradsq.002.bin"

echo "Compare vectors before & after load gradsq - 2 iterations"
check_exit 0 "cmp --quiet $BASE_PREFIX.004.bin $TEST_PREFIX.002.bin"
echo "Compare vectors before & after load gradsq - 4 iterations"
check_exit 0 "cmp --quiet $BASE_PREFIX.006.bin $TEST_PREFIX.004.bin"
echo "Compare vectors before & after load gradsq - final"
check_exit 0 "cmp --quiet $BASE_PREFIX.bin $TEST_PREFIX.bin"

echo "Compare gradsq before & after load gradsq - 2 iterations"
check_exit 0 "cmp --quiet $BASE_PREFIX.gradsq.004.bin $TEST_PREFIX.gradsq.002.bin"
echo "Compare gradsq before & after load gradsq - 4 iterations"
check_exit 0 "cmp --quiet $BASE_PREFIX.gradsq.006.bin $TEST_PREFIX.gradsq.004.bin"
echo "Compare gradsq before & after load gradsq - final"
check_exit 0 "cmp --quiet $BASE_PREFIX.gradsq.bin $TEST_PREFIX.gradsq.bin"

echo "Cleaning up files"
check_exit 0 "rm $BASE_PREFIX.002.bin $BASE_PREFIX.004.bin $BASE_PREFIX.006.bin $BASE_PREFIX.bin"
check_exit 0 "rm $BASE_PREFIX.gradsq.002.bin $BASE_PREFIX.gradsq.004.bin $BASE_PREFIX.gradsq.006.bin $BASE_PREFIX.gradsq.bin"
check_exit 0 "rm $TEST_PREFIX.002.bin $TEST_PREFIX.004.bin $TEST_PREFIX.bin"
check_exit 0 "rm $TEST_PREFIX.gradsq.002.bin $TEST_PREFIX.gradsq.004.bin $TEST_PREFIX.gradsq.bin"
check_exit 0 "rm $VOCAB_FILE $COOCCURRENCE_FILE $COOCCURRENCE_SHUF_FILE"

echo
echo SUMMARY:
if [[ $num_failed -gt 0 ]]; then
  echo $num_failed tests failed.
  exit 1
else
  echo All tests passed.
  exit 0
fi


