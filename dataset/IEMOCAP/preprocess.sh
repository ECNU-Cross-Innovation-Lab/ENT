# stage
stage=0 # start from 0 if you need to start from data preparation
stop_stage=2

# data directory
procdata_dir=$(dirname $0)
dataset_dir="/Dataset/IEMOCAP" # revise this to the directory of the dataset
input_text="$procdata_dir/input.txt"

# bpemode (char or bpe)
train_set=iemocap
nbpe=5000
bpemode=char

. parse_options.sh || exit 1;

if [ ${bpemode} = bpe ]; then
  bpemodel=$procdata_dir/${train_set}_${bpemode}${nbpe}
  dict=$procdata_dir/${train_set}_${bpemode}${nbpe}_units.txt
elif [ ${bpemode} = char ]; then
  dict=$procdata_dir/${train_set}_char_units.txt
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "stage 0: Preprocess Raw Data"
  python $procdata_dir/preprocess.py --prodir $procdata_dir --path $dataset_dir
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  ### Task dependent. You have to check non-linguistic symbols used in the corpus.
  echo "stage 1: Dictionary and Json Data Preparation"
  
  if [ ${bpemode} = bpe ]; then
    echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
    echo "<unk> 1" >> ${dict} # <unk> must be 1

    # we borrowed these code and scripts which are related bpe from ESPnet.
    $procdata_dir/spm_train --input=${input_text} --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    $procdata_dir/spm_encode --model=${bpemodel}.model --output_format=piece < $input_text | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    num_token=$(cat $dict | wc -l)
    echo "<sos/eos> $num_token" >> $dict # <eos>
    wc -l ${dict}
  fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2: Tokenize and rewrite pickle"
  if [ ${bpemode} = bpe ]; then
    python $procdata_dir/toke.py --mode ${bpemode} --data_path $procdata_dir/iemocap4.pkl --out_path $procdata_dir/iemocap4${bpemode}.pkl --symbol_table $dict --bpe_model ${bpemodel}.model
  elif [ ${bpemode} = char ]; then
    python $procdata_dir/toke.py --mode ${bpemode} --data_path $procdata_dir/iemocap4.pkl --out_path $procdata_dir/iemocap4${bpemode}.pkl --symbol_table $dict
  fi
fi