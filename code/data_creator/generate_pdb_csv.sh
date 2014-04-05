PTB_DB_SRC="../../data/ptb_database"
PTB_DB_TARGET="../../data/ptb_database_csv"
START_DIR=$PWD
mkdir -p $PTB_DB_TARGET

while read p; do
	SUBDIR=$(echo $p | grep -o "^[^/]*")
	FILENAME=$(echo $p | grep -o "[^/]*$")
	TARGET_DIR="${START_DIR}/${PTB_DB_TARGET}"
	TARGET_FILE_SAMPLE="${TARGET_DIR}/${FILENAME}"
	SRC_DIR="${START_DIR}/${PTB_DB_SRC}/${SUBDIR}"
	mkdir -p $TARGET_DIR
	cd $SRC_DIR
	#rdsamp -c -s 1 -p -r "${FILENAME}" > "${TARGET_FILE_SAMPLE}.csv"
	$START_DIR/read_headers.pl "${FILENAME}.hea" -c > "${TARGET_FILE_SAMPLE}.descr"
done < RECORDS_small.txt
