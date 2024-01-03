# stage
stage=0 # start from 0 if you need to start from data preparation
stop_stage=0

# data directory
procdata_dir=$(dirname $0) # revise this to the directory of code preprocssing data
dataset_dir="/Dataset/ZED" # revise this to the directory of the dataset
input_text="$procdata_dir/input.txt"


. parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "stage 0: Preprocess Raw Data"
  python $procdata_dir/preprocess.py --prodir $procdata_dir --path $dataset_dir
fi
