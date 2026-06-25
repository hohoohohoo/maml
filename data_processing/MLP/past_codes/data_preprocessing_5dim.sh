OUTPUT_DIR="../../dataset_test5(dim5)_TSMC/processed"
LIB_PREFIX="TSMC_SS_50_"          # .lib 파일 prefix
START_NODE=0
END_NODE=49                        # exclusive
DATA_DIR="../../dataset_tsmc/TSMC_SS_50"
# 실행
python3 build_and_split_dataset_test5dim.py $OUTPUT_DIR $LIB_PREFIX $START_NODE $END_NODE $DATA_DIR
