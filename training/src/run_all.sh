#! /bin/bash
#set x

SECONDS=0
CHK_DIR="/data/home/chenyu/Yangfeiyu/singlePose/log/zq_224/model/yang_mv3_3_2x2_cpm_batch-80_lr-0.001_gpus-4_224x224_..-experiments-zq21_cpm"
MODEL="yang_mv3_3_2x2_cpm"
OUTPUT_NODE_NAME="CPM/stage_2_out"
EXT="meta"
HISTORY=""
LOG_FILE=${CHK_DIR}"/""${OUTPUT_NODE_NAME: -11}"".log"
TMP_FILE=${CHK_DIR}"/.""${OUTPUT_NODE_NAME: -11}"".log"

run_fun(){
/data/home/chenyu/Yangfeiyu/App/anaconda3/envs/tf13/bin/python\
  gen_frozen_pb.py\
  --checkpoint=$1\
  --output_node_names=${OUTPUT_NODE_NAME}\
  --model=${MODEL}\
  &> /dev/null

/data/home/chenyu/Yangfeiyu/App/anaconda3/envs/tf13/bin/python\
  benchmark.py\
  --frozen_pb_path="$1.pb"\
  --output_node_name=${OUTPUT_NODE_NAME}\
  &> $2
}

if [ -e $LOG_FILE ]
then
  echo "[INFO] Read file: $LOG_FILE"
  HISTORY=$(cat $LOG_FILE)
else
  echo "[INFO] Create file: $LOG_FILE"
fi

touch $LOG_FILE

if [ -e $TMP_FILE ]
  then
  rm $TMP_FILE
fi
touch $TMP_FILE

for i in $(ls $CHK_DIR)
do
  if [[ $i =~ $EXT ]]
  then
    CHK_NAME="${i%.*}"

    if [[ $HISTORY =~ $CHK_NAME ]]
    then
	continue
    fi

    CHK_PATH=$CHK_DIR"/"$CHK_NAME

    run_fun $CHK_PATH $TMP_FILE

    LINE1=$(tail -n 1 $TMP_FILE)
    PCKH="${LINE1: -5}"
    
    LINE2=$(tail -n 2 $TMP_FILE | head -n 1)
    TIME="${LINE2: -8:5}"
    
    echo "[INFO] $CHK_NAME $TIME $PCKH"
    echo "$CHK_NAME $TIME $PCKH" >> $LOG_FILE
  fi
done

sort -V -o $LOG_FILE $LOG_FILE
rm $TMP_FILE

echo "+++++ PCKH +++++"
RE='^[0-9]+([.][0-9]+)?$'
LAST_LINE=""
NUM_LINES=$(cat $LOG_FILE | wc -l)

for i in $(seq 1 $NUM_LINES)
do
  LAST_LINE=$(sort -t" " -k3 $LOG_FILE | tail -n $i | head -n 1)
  LAST_WORD=${LAST_LINE##* }

  if [[ $LAST_WORD =~ $RE ]]
  then
    break
  fi
done


while IFS= read -r LINE
do
  if [[ $LINE =~ $LAST_LINE ]]
  then
      echo "$LINE +"
  else
      echo $LINE
  fi
done < $LOG_FILE
echo "++++++++++++++++"


MINIUTES=$((SECONDS / 60))
echo "[INFO] Time elapsed: $MINIUTES min"














