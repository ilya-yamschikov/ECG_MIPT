PTB_DB_SRC="../../data/ptb_database"
PTB_DB_TARGET="../../data/ptb_database_csv"
START_DIR=$PWD
TARGET_DIR="${START_DIR}/${PTB_DB_TARGET}"
INFO_FILE="info.txt"
mkdir -p $PTB_DB_TARGET
rm -f "${TARGET_DIR}/${INFO_FILE}"

while read p; do
	SUBDIR=$(echo $p | grep -o "^[^/]*")
	FILENAME=$(echo $p | grep -o "[^/]*$")
	TARGET_FILE_SAMPLE="${TARGET_DIR}/${FILENAME}"
	SRC_DIR="${START_DIR}/${PTB_DB_SRC}/${SUBDIR}"
	mkdir -p $TARGET_DIR
	cd $SRC_DIR
	rdsamp -c -s 1 -p -r "${FILENAME}" > "${TARGET_FILE_SAMPLE}.csv"
	$START_DIR/read_headers.pl "${FILENAME}.hea" -c > "${TARGET_FILE_SAMPLE}.descr"
	
	# generate file with known classes
	IF_INCLUDE_FILE=$($START_DIR/read_headers.pl "${FILENAME}.hea" -i)
	if [ $IF_INCLUDE_FILE -ne 0 ];
	then
		echo "${FILENAME}"$'\r' >> "${TARGET_DIR}/${INFO_FILE}"
	fi
done < RECORDS.txt
