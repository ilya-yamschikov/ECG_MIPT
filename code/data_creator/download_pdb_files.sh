export PTB_DB_SRC="http://www.physionet.org/physiobank/database/ptbdb/"
export PTB_DB_TARGET="../../data/ptb_database"

while read p; do
	SUBDIR=$(echo $p | grep -o "^[^/]*")
	FILENAME=$(echo $p | grep -o "[^/]*$")
	export SRC_FILE="${PTB_DB_SRC}/${p}"
	export TARGET_DIR="${PTB_DB_TARGET}/${SUBDIR}"
	mkdir -p $TARGET_DIR
	wget -P $TARGET_DIR "${SRC_FILE}.hea"
	wget -P $TARGET_DIR "${SRC_FILE}.dat"
	wget -P $TARGET_DIR "${SRC_FILE}.xyz"
done < RECORDS.txt